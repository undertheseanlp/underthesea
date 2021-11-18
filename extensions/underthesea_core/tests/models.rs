#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn test_crfs(){
        let buf = fs::read("tests/wt_crf_2018_09_13.bin").unwrap();
        let model = crfs::Model::new(&buf).unwrap();
     }
}