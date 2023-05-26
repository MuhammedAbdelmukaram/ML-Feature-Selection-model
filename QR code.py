import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import GappedSquareModuleDrawer
from PIL import Image

# Define the data for the QR code
data = 'https://www.stratflow.io'

# Generate the QR code
qr = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=30,
    border=1,
    version=2
)
qr.add_data(data)

# Create the QR code image with styled modules
img = qr.make_image(image_factory=StyledPilImage, module_drawer=GappedSquareModuleDrawer())

# Open and resize the logo image
logo_path = 'Logo Loader 2.png'  # Replace with the path to your logo image
logo_image = Image.open(logo_path)
logo_image = logo_image.resize((146, 146))  # Adjust the size of the logo as needed

# Calculate the position to place the logo at the center of the QR code
qr_width, qr_height = img.size
logo_width, logo_height = logo_image.size
logo_position = ((qr_width - logo_width) // 2, (qr_height - logo_height) // 2)

# Paste the logo image onto the QR code image
img.paste(logo_image, logo_position)

# Save the final QR code image
img.save('qrCode.png')