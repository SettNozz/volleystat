@echo off
echo 🏐 Volleyball Dataset Curator
echo ===============================
echo.

echo 📊 Checking dataset...
python check_dataset.py

echo.
echo 🚀 Starting curator...
echo 📱 Open http://localhost:8000 in your browser
echo 🎮 Use keyboard: 'A' to Accept, 'S' to Skip
echo.
echo ⚠️  DO NOT CLOSE THIS WINDOW! It will stop the server.
echo.

python run_curator.py 