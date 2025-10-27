
# Configurações de Comparação de Rosto
FACE_COMPARISON = {
    "reference_image": "data/raw/images/elon01.jpg",
    "test_image": "data/raw/images/elon_test.jpg",
}

# Configurações de Contagem de Pessoas
PEOPLE_COUNTING = {
    "video_path": "data/raw/videos/escalator.mp4",
    "roi_coords": (490, 230, 30, 150),  # (x, y, w, h)
    "threshold": 4000,
    "font": "cv2.FONT_HERSHEY_SIMPLEX",
}
