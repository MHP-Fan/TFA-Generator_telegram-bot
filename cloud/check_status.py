# cloud/check_status.py
# Утилита для ручной проверки состояния домашнего сервера из консоли
import requests

# Замените на адрес вашего Flask-приложения на PythonAnywhere
URL = "https://yourusername.pythonanywhere.com/check"

def test_status():
    print("🛰️ Запрос к облачному наблюдателю...")
    try:
        response = requests.get(URL, timeout=10)
        if response.status_code == 200:
            status = response.text
            if "detected offline" in status:
                print("🔴 Статус: ОФЛАЙН. Домашний сервер не подает признаков жизни.")
            else:
                print("🟢 Статус: ОНЛАЙН. Домашний сервер работает стабильно.")
        else:
            print(f"⚠️ Ошибка сервера PythonAnywhere: {response.status_code}")
    except requests.exceptions.Timeout:
        print("❌ Превышено время ожидания ответа от PythonAnywhere.")
    except Exception as e:
        print(f"❌ Не удалось подключиться к облаку: {e}")

if __name__ == "__main__":
    test_status()