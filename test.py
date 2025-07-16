# import fitz  # PyMuPDF

# pdf = fitz.open("16015-ranjana.pdf")
# for page in pdf:
#     fonts = page.get_fonts()
#     for font in fonts:
#         print(font)  # Prints font info (name, embedded status)




# import fitz

# pdf = fitz.open("16015-ranjana.pdf")
# for page in pdf:
#     for font_info in page.get_fonts(full=True):
#         font_ref = font_info[0]
#         font_tuple = pdf.extract_font(font_ref)
#         # Print structure for debugging
#         print([(i, type(x), len(x) if hasattr(x, "__len__") else None) for i, x in enumerate(font_tuple)])
#         font_data = font_tuple[1]  # Usually the font bytes
#         ext = font_tuple[2]        # Usually the extension
#         if isinstance(font_data, bytes):
#             with open(f"{font_ref}{ext}", "wb") as f:
#                 f.write(font_data)
#         else:
#             print(f"Font data for {font_ref} is not bytes!")



from pdf2image import convert_from_path

images = convert_from_path("16015-ranjana.pdf")
for i, img in enumerate(images):
    img.save(f"page_{i}.png")