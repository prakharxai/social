# from PIL import Image, ImageDraw, ImageFont
# import os

# def create_test_image(text, filename="test_image.png"):
#     # Create a white background image with larger dimensions for longer text
#     img = Image.new("RGB", (800, 300), color="white")
#     draw = ImageDraw.Draw(img)
    
#     # Use a default font or specify a path to a .ttf font file
#     try:
#         font = ImageFont.truetype("arial.ttf", 20)
#     except:
#         font = ImageFont.load_default()
    
#     # Draw text with multiline support
#     draw.text((10, 10), text, fill="black", font=font)
    
#     # Save the image
#     img.save(filename)
#     print(f"Test image saved as {filename}")

# # Complex test texts
# complex_texts = [
#     # Social: Positive, nuanced, and community-oriented
#     "In our vibrant community, we strive to foster mutual respect, encourage collaboration, and promote acts of kindness to build a harmonious and supportive environment for all.",
    
#     # Antisocial: Negative, aggressive, and offensive
#     "The toxic behavior of some individuals, filled with hostility and derogatory remarks, undermines our efforts to maintain a respectful and inclusive dialogue.",
    
#     # Neutral: Factual, professional, and objective
#     "The annual conference will feature discussions on technological advancements, with sessions scheduled from 9 AM to 5 PM, covering topics like AI and cybersecurity.",
    
#     # Mixed: Contains both social and antisocial elements
#     "While we aim to support and uplift each other with kindness, some members continue to spread negativity and hostility, creating challenges for our community's harmony.",
    
#     # Complex Social: Positive but with sophisticated language
#     "Through empathetic engagement and unwavering commitment to collective well-being, our group endeavors to cultivate an atmosphere of trust, gratitude, and shared prosperity.",
    
#     # Complex Antisocial: Subtle negativity with strong language
#     "The pervasive undercurrent of disdain and subtle aggression in certain interactions erodes the foundation of civility, fostering an environment of mistrust and discord."
# ]

# # Generate test images
# for i, text in enumerate(complex_texts):
#     create_test_image(text, f"complex_test_image_{i+1}.png")









from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

def create_test_image(text, filename="test_image.png", font_path="arial.ttf", max_width=800, padding=20, font_size=20):
    # Load font (fallback to default if not found)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Wrap text for width
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = wrapper.wrap(text)

    # Create a temporary image to calculate size
    dummy_img = Image.new("RGB", (max_width, 100), color="white")
    draw = ImageDraw.Draw(dummy_img)

    # Calculate total height based on lines
    total_text_height = 0
    for line in wrapped_text:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = (bbox[3] - bbox[1]) + 5  # line height + spacing
        total_text_height += line_height

    image_height = padding * 2 + total_text_height

    # Create final image with proper height
    img = Image.new("RGB", (max_width, image_height), color="white")
    draw = ImageDraw.Draw(img)

    y_text = padding
    for line in wrapped_text:
        draw.text((padding, y_text), line, font=font, fill="black")
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = (bbox[3] - bbox[1]) + 5
        y_text += line_height

    # Save image
    img.save(filename)
    print(f"✅ Saved: {filename}")

# ------------------------------
# Original social and mixed texts
original_texts = [
    "In our vibrant community, we strive to foster mutual respect, encourage collaboration, and promote acts of kindness to build a harmonious and supportive environment for all.",
    "The toxic behavior of some individuals, filled with hostility and derogatory remarks, undermines our efforts to maintain a respectful and inclusive dialogue.",
    "The annual conference will feature discussions on technological advancements, with sessions scheduled from 9 AM to 5 PM, covering topics like AI and cybersecurity.",
    "While we aim to support and uplift each other with kindness, some members continue to spread negativity and hostility, creating challenges for our community's harmony.",
    "Through empathetic engagement and unwavering commitment to collective well-being, our group endeavors to cultivate an atmosphere of trust, gratitude, and shared prosperity.",
    "The pervasive undercurrent of disdain and subtle aggression in certain interactions erodes the foundation of civility, fostering an environment of mistrust and discord."
]

# Complex antisocial test cases
extra_antisocial_texts = [
    "It's astonishing how some people manage to contribute absolutely nothing while pretending to be the backbone of this team. Their incompetence is both predictable and exhausting.",
    "If this nonsense keeps happening, I won’t be responsible for what I say next. People need to be taught a lesson — permanently.",
    "Every time she opens her mouth, I lose a few brain cells. It’s painful watching incompetence parade as leadership.",
    "If you took things less personally and more seriously, maybe we wouldn’t have to clean up your mess all the time. But I guess accountability is too much to ask.",
    "It’s impressive how you’ve managed to lower the bar even further. I didn’t think it was possible, but here we are.",
    "People like you are the reason no one wants to work here. The constant whining, the finger-pointing — it’s pathetic.",
    "What masquerades as constructive feedback often turns out to be a façade for mediocrity wrapped in arrogance and cloaked in defensiveness.",
    "This entire department thrives on favoritism, ego, and blame-shifting. No one’s actually solving problems — just creating new ones and pointing fingers.",
    "Some individuals think that showing up occasionally gives them the right to undermine others who actually care about the work.",
    "Your opinion is noted, albeit as yet another example of unverified speculation dressed up as insight. Next."
]

# Combine all texts
all_texts = original_texts + extra_antisocial_texts

# Output folder
output_folder = "test_images"
os.makedirs(output_folder, exist_ok=True)

# Generate images
for i, text in enumerate(all_texts):
    filename = os.path.join(output_folder, f"complex_test_image_{i+1}.png")
    create_test_image(text, filename)
