use paddleknife_core::image_io;
use std::path::PathBuf;

fn sample_png() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hello_world.png")
}

#[test]
fn loads_png_as_single_page() {
    let page = image_io::load_image(&sample_png()).expect("load image");
    assert_eq!(page.index, 0);
    assert!(page.width > 0);
    assert!(page.height > 0);
}
