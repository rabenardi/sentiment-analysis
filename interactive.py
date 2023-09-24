from model_test import predict_sentiment

# Mengambil input user dan memprediksinya
while True:
    sentence = str(input("Masukkan kalimat: "))
    print(predict_sentiment(sentence))