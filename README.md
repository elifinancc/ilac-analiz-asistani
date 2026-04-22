#  İlaç Analiz Asistanı

Yapay zeka (AI) destekli bu uygulama, yüklediğiniz veya kameranızla çektiğiniz ilaç kutusu fotoğraflarını analiz ederek size detaylı ve bilimsel bir ilaç rehberi sunar. 

Google Gemini ve Groq (Llama 3) AI modelleri kullanılarak geliştirilmiş olup, kolay arayüzü sayesinde ilaçların etken maddeleri, kullanım alanları, yan etkileri ve dikkat edilmesi gereken noktalar hakkında anında detaylı rapor almanızı sağlar.

##  Özellikler

- **Görselden İlaç Tanıma:** İlaç kutusunun fotoğrafından (OCR ve Gemini modelini kullanarak) ilacın adını ve etken maddesini otomatik tespit etme.
- ** Otomatik Web Araması:** İlaç hakkında güncel prospektüs verilerini tarama (DuckDuckGo arama entegrasyonu).
- ** Akıllı Raporlama:** Groq altyapısındaki güçlü yapay zeka ile ilacın endikasyonları, yan etkileri ve muadillerini içeren yapılandırılmış bir rapor oluşturma.
- ** Dışa Aktarma:** Analiz edilen sonuçları **PDF** veya **TXT** formatında bilgisayarınıza indirebilme.
- ** Mobil Uyumlu Arayüz:** Streamlit ile sağlanan şık, basit ve mobil cihazlarla da uyumlu kullanıcı arayüzü.

## Kullanılan Teknolojiler

- **Python & Streamlit:** Web arayüzü ve arka plan mantığı
- **Google Gemini API:** Görüntü işleme ve veriyi ayıklama
- **Groq API (Llama-3.3-70b):** Analiz ve medikal metin sentezi
- **EasyOCR:** Görselden genel metin tanıma (Yedek sistem)
- **FPDF:** Raporları PDF'e dönüştürme

