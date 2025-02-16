use std::path::Path;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use curve25519_dalek::{
    constants::RISTRETTO_BASEPOINT_POINT as G,
    ecdlp::{
        self,
        ECDLPArguments,
        ECDLPTables
    },
    Scalar,
};
use rand::Rng;

pub fn ecdlp_bench(c: &mut Criterion) {
    if !Path::new("ecdlp_table.bin").exists() {
        let tables = ECDLPTables::generate(26).unwrap();
        tables.write_to_file("ecdlp_table.bin").unwrap();
    }

    let tables = ECDLPTables::load_from_file(26, "ecdlp_table.bin").unwrap();
    let view = tables.view();

    c.bench_function("fast ecdlp", |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48).pseudo_constant_time(true),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=1", |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(1),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=2", |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(2),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=4", |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(4),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=8", |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(4),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("fast ecdlp find 0", |b| {
        let num: u64 = 0;
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 47)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 47));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {} T=2", (1i64 << 47)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 47));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48).n_threads(1),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 44)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 44));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 43)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 43));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 26)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 26));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 27)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 27));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });
}

criterion_group!(ecdlp, ecdlp_bench);
criterion_main!(ecdlp);
