from PIL import Image,  ImageDraw


def marker_function(image, x1, x2, y1, y2, leye, reye):
    """ This function just mark the face and its components with a green line """

    mask_clr = 'lawngreen'
    mask_width = 1
    draw = ImageDraw.Draw(image)
    draw.rectangle(((x1, y1), (x2, y2)), outline=mask_clr, width=mask_width)

    if (leye is None or reye is None):
        image = image.convert("RGB")

        image.save('marked.jpeg')
        return image
    draw.line((x1, y1)+leye, fill=mask_clr, width=mask_width)
    draw.line((x2, y2)+reye, fill=mask_clr, width=mask_width)
    draw.line((x2, y1)+reye, fill=mask_clr, width=mask_width)
    draw.line((x1, y2)+leye, fill=mask_clr, width=mask_width)
    draw.line((x2, y1)+leye, fill=mask_clr, width=mask_width)
    draw.line((x1, y1)+reye, fill=mask_clr, width=mask_width)
    draw.line((x2, (y1+y2)//2)+reye, fill=mask_clr, width=mask_width)
    draw.line((x1, (y1+y2)//2)+leye, fill=mask_clr, width=mask_width)
    draw.line(((x1+x2)//2, y1)+reye, fill=mask_clr, width=mask_width)
    draw.line(((x1+x2)//2, y1)+leye, fill=mask_clr, width=mask_width)

    draw.line(leye+reye, fill=mask_clr, width=mask_width)

    image.save('marked.jpeg')
