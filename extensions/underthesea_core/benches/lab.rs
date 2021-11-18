extern crate rayon;
extern crate criterion;

use std::time::Duration;
use rayon::prelude::*;
use criterion::*;
use underthesea_core;

fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 1,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn f1() {
    let mut arr: [u32; 2] = [10, 100];
    arr.par_iter_mut().map(|n| fibonacci(*n));
    return;
}

fn f2() {
    let mut arr: [u32; 2] = [10, 100];
    arr.map(|n| fibonacci(n));
    return;
}

fn f3() {
    let a = 1 + 2;
    return;
}


fn criterion_benchmark(c: &mut Criterion){
    let mut group = c.benchmark_group("abc");
    group.bench_function("my-function", |b| b.iter(|| f3()));
    group.bench_function("my-function-2", |b| b.iter(|| f2()));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = criterion_benchmark
}

criterion_main!(benches);