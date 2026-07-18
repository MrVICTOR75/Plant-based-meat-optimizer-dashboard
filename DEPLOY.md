# คู่มือ Deploy แดชบอร์ดขึ้น Streamlit Community Cloud (ฟรี, ลิงก์ถาวร)

ผลลัพธ์: แดชบอร์ดจะได้ **URL ถาวร** เช่น `https://plant-meat-optimizer.streamlit.app`
ที่เปิดได้ตลอด แชร์ได้ และไม่เปลี่ยนทุกครั้งที่รัน (ต่างจาก localhost หรือ ngrok)

---

## 0) เตรียมไว้ให้แล้ว (ในโฟลเดอร์นี้)
- `main_inpho.py` — ไฟล์เริ่มของแอป (entry point)
- `requirements.txt` — รายการไลบรารี (streamlit, pandas, numpy, scipy, plotly, openpyxl)
- `.gitignore` — กันไฟล์ `*.zip` ก้อนใหญ่ไม่ให้ขึ้น GitHub (เกินขีดจำกัด 100 MB/ไฟล์)
- `Nutrition_Matrix.xlsx` — ตารางโภชนาการจริง (5 KB) ที่ commit ขึ้นได้
  บนคลาวด์แอปจะโหลดไฟล์นี้เป็น "ข้อมูลจริง" และรีเฟรชคอลัมน์ต้นทุนจาก `config.py` อัตโนมัติ
  (จึงไม่ต้องอัปโหลดไฟล์ FoodData zip ก้อนใหญ่)

## 1) สร้าง GitHub repo แล้ว push โค้ด
```bash
cd "path/ไปยังโฟลเดอร์ Code นี้"
git init
git add .
git commit -m "Plant-based meat optimizer dashboard"
git branch -M main
git remote add origin https://github.com/<ชื่อผู้ใช้>/<ชื่อ-repo>.git
git push -u origin main
```
> ไฟล์ `*.zip` จะถูกข้าม (ตาม `.gitignore`) โดยอัตโนมัติ — แอปบนคลาวด์ใช้ `Nutrition_Matrix.xlsx` แทน

## 2) Deploy บน Streamlit Community Cloud
1. ไปที่ **https://share.streamlit.io** แล้ว **Sign in with GitHub** (ครั้งแรกให้กด Authorize)
2. กด **Create app** → **Deploy a public app from GitHub**
3. กรอก:
   - **Repository**: `<ชื่อผู้ใช้>/<ชื่อ-repo>`
   - **Branch**: `main`
   - **Main file path**: `main_inpho.py`
   - **App URL (subdomain)**: ตั้งชื่อที่จำง่าย เช่น `plant-meat-optimizer`
4. กด **Deploy** แล้วรอ 2–5 นาที
5. ได้ลิงก์ถาวร: `https://plant-meat-optimizer.streamlit.app`

## 3) การอัปเดตภายหลัง
แก้โค้ดในเครื่อง แล้ว `git push` — Streamlit Cloud จะ redeploy ให้เองอัตโนมัติ (ลิงก์เดิมไม่เปลี่ยน)

---

## 4) ใส่ลิงก์ในรายงาน (LaTeX)

**แบบไฮเปอร์ลิงก์** (ต้องมี `\usepackage{hyperref}`):
```latex
ผู้สนใจสามารถทดลองใช้แดชบอร์ดออนไลน์ได้ที่
\href{https://plant-meat-optimizer.streamlit.app}{plant-meat-optimizer.streamlit.app}
```

**แบบ QR code** ให้ผู้อ่านฉบับพิมพ์สแกน (ต้องมี `\usepackage{qrcode}`):
```latex
\begin{figure}[H]
\centering
\qrcode[height=3cm]{https://plant-meat-optimizer.streamlit.app}
\caption{สแกนเพื่อเปิดแดชบอร์ดออนไลน์}
\end{figure}
```
> เปลี่ยน `plant-meat-optimizer` เป็น subdomain จริงที่ตั้งไว้ในขั้นที่ 2

---

## หมายเหตุ
- บนคลาวด์แอปใช้ **ข้อมูลโภชนาการจริง** จาก `Nutrition_Matrix.xlsx` (คำนวณไว้แล้ว) + ต้นทุนจาก `config.py`
  ถ้าต้องการให้แอป "สร้างตารางใหม่จากไฟล์ USDA ดิบ" ต้องมีไฟล์ zip ในเครื่อง (ทำได้เฉพาะรันในเครื่อง ไม่ใช่บนคลาวด์)
- แผนฟรีของ Streamlit Cloud เหมาะกับงานสาธิต/รายงาน หากแอปไม่มีผู้เข้าใช้งานสักพัก อาจเข้าสู่โหมดพัก (sleep) แล้วปลุกกลับมาได้เมื่อเปิดลิงก์อีกครั้ง
