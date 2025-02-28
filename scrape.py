import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# === SETUP WEBDRIVER TANPA DOWNLOAD CHROMEDRIVER MANUAL ===
options = webdriver.ChromeOptions()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Inisialisasi WebDriver dengan WebDriver Manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# === SETUP SEARCH QUERY DAN FOLDER PENYIMPANAN ===
search_query = "Japanese bamboo leaves"
save_folder = "./newdata/bambujepang"

# Buat folder jika belum ada
os.makedirs(save_folder, exist_ok=True)

# === SCRAPE GOOGLE IMAGES ===
driver.get(f"https://www.google.com/search?tbm=isch&q={search_query}")

# Scroll ke bawah untuk load lebih banyak gambar
for _ in range(5):
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(2)

# Ambil elemen gambar
images = driver.find_elements(By.CSS_SELECTOR, "img")

# Filter hanya gambar dengan format yang valid
valid_formats = (".png", ".jpg", ".jpeg")
image_urls = [img.get_attribute("src") for img in images if img.get_attribute("src") and img.get_attribute("src").lower().endswith(valid_formats)]

# === DOWNLOAD GAMBAR ===
for idx, img_url in enumerate(image_urls):
    try:
        # Tentukan ekstensi file
        ext = ".png" if img_url.endswith(".png") else ".jpg"
        filename = f"bambujepang{idx+1}{ext}"  # Nama file berurutan
        file_path = os.path.join(save_folder, filename)

        # Download gambar
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"[✔] Downloaded: {filename}")
        else:
            print(f"[✘] Failed: {img_url}")

    except Exception as e:
        print(f"[!] Error downloading {img_url}: {e}")

# Tutup browser
driver.quit()
print("\n✅ Semua gambar berhasil di-download!")
