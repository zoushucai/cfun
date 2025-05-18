from cfun.yolo.makechar import MakeCharImage


def test_makechar():
    generator = MakeCharImage(
        text="好",
        image_size=(64, 64),
        offset=0.5,
        font_path=None,
        output_path="output/AB.png",
        noise_density=0.25,
    )
    generator.generate_image()
    generator.save_image()


if __name__ == "__main__":
    test_makechar()
