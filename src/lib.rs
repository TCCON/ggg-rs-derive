use darling::{FromDeriveInput, FromField, ast};
use proc_macro::TokenStream;
use quote::quote;
use syn::{GenericParam, Type};

#[derive(FromField)]
#[darling(attributes(subset))]
struct SubsetField {
    ident: Option<syn::Ident>,
    ty: syn::Type,

    #[darling(default)]
    skip: bool,

    #[darling(default)]
    kind: Option<String>,

    #[darling(default)]
    axis: Option<usize>,
}

#[derive(FromDeriveInput)]
#[darling(attributes(subset), supports(struct_named))]
struct SubsetInput {
    ident: syn::Ident,
    generics: syn::Generics,
    data: ast::Data<darling::util::Ignored, SubsetField>,
}

enum ValueKind {
    Vec,
    NdArray { axis: usize },
}

enum FieldKind {
    Vec,
    NdArray { axis: usize },
    HashMap { value: ValueKind },
    Skip,
}

/// Given a `Vec<T>` or `Array*<T>` type, classify it. Reused for both
/// top-level fields and HashMap values.
fn classify_container(
    ty: &Type,
    explicit_kind: Option<&str>,
    explicit_axis: Option<usize>,
) -> Result<ValueKind, darling::Error> {
    if let Some(kind) = explicit_kind {
        return match kind {
            "vec" => Ok(ValueKind::Vec),
            "array" => Ok(ValueKind::NdArray {
                axis: explicit_axis.unwrap_or(0),
            }),
            other => Err(darling::Error::custom(format!("unknown kind \"{other}\""))),
        };
    }
    if let Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            let ident = seg.ident.to_string();
            if ident == "Vec" {
                return Ok(ValueKind::Vec);
            }
            if ident.starts_with("Array") {
                return Ok(ValueKind::NdArray {
                    axis: explicit_axis.unwrap_or(0),
                });
            }
        }
    }
    Err(darling::Error::custom("couldn't infer container kind"))
}

/// Extract the `V` type param out of `HashMap<K, V>`.
fn hashmap_value_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty {
        let seg = type_path.path.segments.last()?;
        if seg.ident != "HashMap" {
            return None;
        }
        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
            // args: [K, V] — value is the second generic argument.
            if let Some(syn::GenericArgument::Type(v_ty)) = args.args.iter().nth(1) {
                return Some(v_ty);
            }
        }
    }
    None
}

fn resolve_kind(f: &SubsetField) -> Result<FieldKind, darling::Error> {
    let ident = f.ident.as_ref().unwrap();

    if f.skip {
        if f.kind.is_some() || f.axis.is_some() {
            return Err(
                darling::Error::custom("`skip` cannot be combined with `kind` or `axis`")
                    .with_span(ident),
            );
        }
        return Ok(FieldKind::Skip);
    }

    if let Some(kind) = &f.kind {
        return match kind.as_str() {
            "array" => Ok(FieldKind::NdArray {
                axis: f.axis.unwrap_or(0),
            }),
            "vec" => {
                if f.axis.is_some() {
                    return Err(darling::Error::custom(
                        "`axis` has no effect on `kind = \"vec\"` fields",
                    )
                    .with_span(ident));
                }
                Ok(FieldKind::Vec)
            }
            other => Err(darling::Error::custom(format!(
                "unknown kind \"{other}\", expected \"vec\" or \"array\""
            ))
            .with_span(ident)),
        };
    }

    // Handle HashMaps of vecs/arrays
    if let Some(value_ty) = hashmap_value_type(&f.ty) {
        let value_kind = classify_container(value_ty, f.kind.as_deref(), f.axis)
            .map_err(|e| e.with_span(ident))?;
        return Ok(FieldKind::HashMap { value: value_kind });
    }

    // Fallback: infer from the type's last path segment.
    if let Type::Path(type_path) = &f.ty {
        if let Some(seg) = type_path.path.segments.last() {
            let ty_ident = seg.ident.to_string();
            if ty_ident == "Vec" {
                if f.axis.is_some() {
                    return Err(darling::Error::custom(
                        "`axis` has no effect on inferred Vec fields",
                    )
                    .with_span(ident));
                }
                return Ok(FieldKind::Vec);
            }
            if ty_ident.starts_with("Array") {
                return Ok(FieldKind::NdArray {
                    axis: f.axis.unwrap_or(0),
                });
            }
        }
    }

    Err(darling::Error::custom(
        "couldn't infer field kind — annotate with #[subset(kind = \"vec\")], \
         #[subset(kind = \"array\")], or #[subset(skip)]",
    )
    .with_span(ident))
}

/// Adds a `.subset(&[usize])` method to a struct to subset [`Vec`] or ndarray Array fields by indices.
///
/// In basic usage, annotating a structure with `#[derive(Subset)]` will implement the `.subset`
/// method. This requires that all fields be `Vec` or `ndarray` Array types. Specifically, the
/// types must be `Vec`, start with `Array`, or start with `ndarray::Array`. The default `.subset`
/// method will then subset multi-dimensional arrays along their first dimension:
///
/// ```rust
/// use ggg_rs_derive::Subset;
/// use ndarray::{Array1, array};
///
/// #[derive(Subset)]
/// struct Data {
///     v: Vec<i32>,
///     a1: Array1<f32>,
///     a2: ndarray::Array2<i64>
/// }
///
/// let orig = Data {
///     v: vec![100, 200, 300],
///     a1: array![10.0, 20.0, 30.0],
///     a2: array![[1, -1], [2, -2], [3, -3]]
/// };
///
/// let sub = orig.subset(&[0, 2]);
/// assert_eq!(sub.v, vec![100, 300]);
/// assert_eq!(sub.a1, array![10., 30.]);
/// assert_eq!(sub.a2, array![[1, -1], [3, -3]]);
/// ```
///
/// Note that `.subset()` will panic if any of the indices are out of bounds
/// for any of the fields.
///
/// # Field attributes
///
/// `subset(skip)` allows you to skip subsetting fields, either because they cannot
/// or should not be subset:
///
/// ```rust
/// use ggg_rs_derive::Subset;
///
/// #[derive(Subset)]
/// struct Data {
///     v: Vec<i32>,
///     #[subset(skip)]
///     s: &'static str
/// }
///
/// let orig = Data { v: vec![1, 2], s: "Hello, world!" };
/// let sub = orig.subset(&[1]);
/// assert_eq!(sub.v, vec![2]);
/// assert_eq!(sub.s, "Hello, world!");
/// ```
///
/// `subset(kind = "vec|array")` allows you to indicate whether the field should be subset
/// as a `Vec` or `ndarray::Array*`. This is necessary if, for example, you have a type
/// alias:
///
/// ```rust
/// use ggg_rs_derive::Subset;
/// use ndarray::array;
///
/// type NVector = Vec<f64>;
/// type Matrix = ndarray::Array2<f64>;
///
/// #[derive(Subset)]
/// struct Data {
///     #[subset(kind = "vec")]
///     v: NVector,
///
///     #[subset(kind = "array")]
///     a: Matrix
/// }
///
/// let orig = Data {
///     v: vec![100.0, 200.0],
///     a: array![[1., 10.], [2., 20.]]
/// };
/// let sub = orig.subset(&[0]);
///
/// assert_eq!(sub.v, vec![100.0]);
/// assert_eq!(sub.a, array![[1.0, 10.0]]);
/// ```
///
/// `subset(axis = i)` allows you to subset along a different axis of an
/// `ndarray` than the first:
///
/// ```rust
/// use ggg_rs_derive::Subset;
/// use ndarray::{Array2, array};
///
/// #[derive(Subset)]
/// struct Data {
///     #[subset(axis = 1)]
///     a: Array2<i32>
/// }
///
/// let orig = Data { a: array![[1, 2, 3, 4], [10, 20, 30, 40]] };
/// let sub = orig.subset(&[1, 3]);
/// assert_eq!(sub.a, array![[2, 4], [20, 40]]);
/// ```
#[proc_macro_derive(Subset, attributes(subset))]
pub fn derive_subset(input: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input as syn::DeriveInput);
    let parsed = match SubsetInput::from_derive_input(&derive_input) {
        Ok(p) => p,
        Err(e) => return e.write_errors().into(),
    };

    let name = &parsed.ident;
    let fields = parsed
        .data
        .take_struct()
        .expect("darling's supports(struct_named) guarantees this")
        .fields;

    let mut vec_idents = Vec::new();
    let mut array_idents = Vec::new();
    let mut array_axes = Vec::new();
    let mut hashmap_vec_idents = Vec::new();
    let mut hashmap_array_idents = Vec::new();
    let mut hashmap_array_axes = Vec::new();
    let mut skip_idents = Vec::new();
    let mut errors = darling::Error::accumulator();

    for f in &fields {
        if let Some(kind) = errors.handle(resolve_kind(f)) {
            let ident = f.ident.clone().unwrap();
            match kind {
                FieldKind::Vec => vec_idents.push(ident),
                FieldKind::NdArray { axis } => {
                    array_idents.push(ident);
                    array_axes.push(axis);
                }
                FieldKind::HashMap {
                    value: ValueKind::Vec,
                } => hashmap_vec_idents.push(ident),
                FieldKind::HashMap {
                    value: ValueKind::NdArray { axis },
                } => {
                    hashmap_array_idents.push(ident);
                    hashmap_array_axes.push(axis);
                }
                FieldKind::Skip => skip_idents.push(ident),
            }
        }
    }

    // Emit every field-level error at once instead of stopping at the first.
    if let Err(e) = errors.finish() {
        return e.write_errors().into();
    }

    let mut generics = parsed.generics.clone();
    for param in generics.params.iter_mut() {
        if let GenericParam::Type(type_param) = param {
            type_param.bounds.push(syn::parse_quote!(Clone));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            fn subset(&self, indices: &[usize]) -> Self {
                Self {
                    #(
                        #vec_idents: indices.iter().map(|&i| self.#vec_idents[i].clone()).collect(),
                    )*
                    #(
                        #array_idents: self.#array_idents.select(::ndarray::Axis(#array_axes), indices),
                    )*
                    #(
                        #hashmap_vec_idents: self.#hashmap_vec_idents
                            .iter()
                            .map(|(k, v)| (
                                k.clone(),
                                indices.iter().map(|&i| v[i].clone()).collect(),
                            ))
                            .collect(),
                    )*
                    #(
                        #hashmap_array_idents: self.#hashmap_array_idents
                            .iter()
                            .map(|(k, v)| (
                                k.clone(),
                                v.select(::ndarray::Axis(#hashmap_array_axes), indices),
                            ))
                            .collect(),
                    )*
                    #(
                        #skip_idents: self.#skip_idents.clone(),
                    )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
