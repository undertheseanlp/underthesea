extern crate rayon;
extern crate criterion;
extern crate underthesea_core;

use std::time::Duration;
use rayon::prelude::*;
use criterion::*;
use std::collections::HashSet;

use underthesea_core::featurizers::CRFFeaturizer;


fn bench_featurizers() {
    let feature_configs = vec![
        "T[0]".to_string(),
        "T[0].is_in_dict".to_string()
    ];
    let mut dictionary = HashSet::new();
    dictionary.insert("giành".to_string());
    dictionary.insert("quả".to_string());
    dictionary.insert("bóng".to_string());
    let new_featurizer = CRFFeaturizer::new(feature_configs, dictionary);
    let sentences = vec![
            vec![
                vec!["Messi".to_string(), "X".to_string()],
                vec!["giành".to_string(), "X".to_string()],
                vec!["quả".to_string(), "X".to_string()],
                vec!["Bóng".to_string(), "X".to_string()],
                vec!["Đá".to_string(), "X".to_string()],
                vec!["1".to_string(), "X".to_string()],
            ]
        ];
    new_featurizer.process(sentences);
    return;
}


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("featurizers", |b| b.iter(|| bench_featurizers()));
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(1));
    targets = criterion_benchmark
}

criterion_main!(benches);