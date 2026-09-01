use std::{assert_eq, collections::HashMap};

use ggg_rs_derive::Subset;
use ndarray::{Array1, Array2, array};

#[derive(Debug, Subset)]
struct AllMaps {
    vec_map: HashMap<&'static str, Vec<i32>>,
    arr1_map: HashMap<&'static str, Array1<i32>>,
    arr2_map: HashMap<&'static str, Array2<i32>>,
}

#[test]
fn subset_maps() {
    let m = AllMaps {
        vec_map: HashMap::from_iter([("a", vec![100, 200, 300]), ("b", vec![400, 500, 600])]),
        arr1_map: HashMap::from_iter([("c", array![10, 20, 30]), ("d", array![40, 50, 60])]),
        arr2_map: HashMap::from_iter([
            ("e", array![[1, -1], [2, -2], [3, -3]]),
            ("f", array![[4, -4], [5, -5], [6, -6]]),
        ]),
    };
    let sub = m.subset(&[0, 2]);
    assert_eq!(
        sub.vec_map,
        HashMap::from_iter([("a", vec![100, 300]), ("b", vec![400, 600])])
    );
    assert_eq!(
        sub.arr1_map,
        HashMap::from_iter([("c", array![10, 30]), ("d", array![40, 60])])
    );
    assert_eq!(
        sub.arr2_map,
        HashMap::from_iter([
            ("e", array![[1, -1], [3, -3]]),
            ("f", array![[4, -4], [6, -6]]),
        ])
    )
}

#[derive(Debug, Subset)]
struct SkipMap {
    vec_map: HashMap<&'static str, Vec<i32>>,
    #[subset(skip)]
    skip_map: HashMap<&'static str, u32>,
}

#[test]
fn subset_maps_skip() {
    let m = SkipMap {
        vec_map: HashMap::from_iter([("a", vec![100, 200, 300]), ("b", vec![400, 500, 600])]),
        skip_map: HashMap::from_iter([("alpha", 0), ("bravo", 1)]),
    };
    let sub = m.subset(&[0, 2]);
    assert_eq!(
        sub.vec_map,
        HashMap::from_iter([("a", vec![100, 300]), ("b", vec![400, 600])])
    );
    assert_eq!(sub.skip_map, m.skip_map);
}
