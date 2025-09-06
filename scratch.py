from PIL import Image

if __name__ == "__main__":
    print("bruh")

    image = Image.open("dataset/0/0.png")
    image = image.convert("RGBA")

    pix00 = image.getpixel((0,0))
    print(f"{pix00=}")

    for i in range(image.height):
        for j in range(image.height):
            pixel = image.getpixel((i,j))
            print(f"{pixel[3]:.1f} ", end="")
        print()
    