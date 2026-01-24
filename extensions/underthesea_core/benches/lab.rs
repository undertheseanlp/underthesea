#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_must_use)]
#![allow(clippy::needless_return)]
#![allow(clippy::redundant_closure)]

extern crate criterion;
extern crate rayon;

use criterion::*;
use rayon::prelude::*;
use std::time::Duration;

fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 1,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn f1() {
    let mut arr: [u32; 2] = [10, 100];
    let _ = arr.par_iter_mut().map(|n| fibonacci(*n));
}

fn f2() {
    let arr: [u32; 2] = [10, 100];
    let _ = arr.map(fibonacci);
}

fn f3() {
    let _a = 1 + 2;
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("abc");
    group.bench_function("my-function", |b| b.iter(f3));
    group.bench_function("my-function-2", |b| b.iter(f2));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = criterion_benchmark
}

criterion_main!(benches);
