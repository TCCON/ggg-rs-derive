use ggg_rs_derive::Subset;
use ndarray::{Array1, Array2, array};

#[derive(Subset, Debug, PartialEq)]
struct Dataset {
    ids: Vec<u64>,
    arr1d: Array1<f64>,
    arr2d: ndarray::Array2<f64>,
    #[subset(skip)]
    name: String,
}

#[test]
fn subsets_vec_and_array_fields() {
    let d = Dataset {
        ids: vec![10, 20, 30],
        arr1d: array![10.0, 20.0, 30.0],
        arr2d: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        name: "test".into(),
    };

    let sub = d.subset(&[0, 2]);

    assert_eq!(sub.ids, vec![10, 30]);
    assert_eq!(sub.arr1d, array![10.0, 30.0]);
    assert_eq!(sub.arr2d, array![[1.0, 2.0], [5.0, 6.0]]);
    assert_eq!(sub.name, "test"); // skipped field cloned whole
}

#[test]
fn empty_indices_yields_empty_fields() {
    let d = Dataset {
        ids: vec![1, 2, 3],
        arr1d: array![10.0, 20.0, 30.0],
        arr2d: array![[1.0], [2.0], [3.0]],
        name: "x".into(),
    };
    let sub = d.subset(&[]);
    assert!(sub.ids.is_empty());
    assert_eq!(sub.arr1d.shape(), &[0]);
    assert_eq!(sub.arr2d.shape(), &[0, 1]);
}

#[test]
#[should_panic]
fn out_of_bounds_index_panics() {
    let d = Dataset {
        ids: vec![1, 2],
        arr1d: array![10.0, 20.0],
        arr2d: array![[1.0], [2.0]],
        name: "x".into(),
    };
    d.subset(&[5]);
}

type Sneaky = Vec<i32>;

// Index different axis of 2D array and override vec vs. array
#[derive(Subset, Debug, PartialEq)]
struct UsingOptions {
    #[subset(kind = "vec")]
    v: Sneaky,
    #[subset(axis = 1)]
    arr: Array2<i32>,
}

#[test]
fn test_field_options() {
    let orig = UsingOptions {
        v: vec![1, 2, 3],
        arr: array![[10, 20, 30], [100, 200, 300]],
    };

    let sub = orig.subset(&[0, 2]);
    assert_eq!(sub.v, vec![1, 3]);
    assert_eq!(sub.arr, array![[10, 30], [100, 300]]);
}

// Generic struct case
#[derive(Subset, Debug, PartialEq)]
struct Generic<T: Clone> {
    values: Vec<T>,
    arr1d: Array1<T>,
    arr2d: Array2<T>,
}

#[test]
fn works_with_generics() {
    let g = Generic {
        values: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        arr1d: array!["A".to_string(), "B".to_string(), "C".to_string()],
        arr2d: array![
            ["a".to_string(), "A".to_string()],
            ["b".to_string(), "B".to_string()],
            ["c".to_string(), "C".to_string()]
        ],
    };
    let sub = g.subset(&[1]);
    assert_eq!(sub.values, vec!["b".to_string()]);
    assert_eq!(sub.arr1d, array!["B".to_string()]);
    assert_eq!(sub.arr2d, array![["b".to_string(), "B".to_string()]]);
}
