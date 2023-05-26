from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Sunucu adresi ve portu
server_address = ('', 8000)


# Sunucu davranışını tanımlayan özel bir sınıf oluşturma
class RequestHandler(BaseHTTPRequestHandler):


    def do_GET(self):
        # Ana sayfa işleme
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            # Ana sayfa içeriği
            content = '''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Basit Hesaplama Sitesi</title>
            </head>
            <body>
                <h1>Hesaplama Formu</h1>
                <form id="hesaplama-formu" onsubmit="hesaplamaYap(); return false;">
                    <label for="age">Yaş:</label>
                    <input type="number" id="age" required><br><br>

                    <label for="job">job:</label>
                    <select id="job" required>
                        <option value="technician">technician</option>
                        <option value="blue-collar">blue-collar</option>
                        <option value="admin">admin.</option>
                        <option value="retired">retired</option>
                        <option value="self-employed">self-employed</option>
                        <option value="student">student</option>
                        <option value="unemployed">unemployed</option>
                        <option value="entrepreneur">entrepreneur</option>
                        <option value="services">services</option>
                        <option value="housemaid">housemaid</option>
                        <option value="management">management</option>
                        <option value="unknown">unknown</option>
                    </select><br><br>

                    <label for="marital">marital:</label>
                    <select id="marital" required>
                        <option value="married">married</option>
                        <option value="single">single</option>
                        <option value="divorced">divorced</option>
                        <!-- Diğer medeni durum seçenekleri -->
                    </select><br><br>
                    <label for="education">education:</label>
                    <select id="education" required>
                        <option value="primary">primary</option>
                        <option value="secondary">secondary</option>
                        <option value="tertiary">tertiary</option>
                        <option value="unknown">unknown</option>
                    </select><br><br>
                    <label for="default">default:</label>
                    <select id="default" required>
                        <option value="yes">yes</option>
                        <option value="no">no</option>
                    </select><br><br>

                    <label for="balance">balance:</label>
                    <input type="number" id="balance" required><br><br>

                    <label for="housing">housing:</label>
                    <select id="housing" required>
                        <option value="yes">yes</option>
                        <option value="no">no</option>
                    </select><br><br>

                    <label for="loan">loan:</label>
                    <select id="loan" required>
                        <option value="yes">yes</option>
                        <option value="no">no</option>
                    </select><br><br>

                    <label for="contact">contact:</label>
                    <select id="contact" required>
                        <option value="cellular">cellular</option>
                        <option value="telephone">telephone</option>
                        <option value="unknown">unknown</option>
                    </select><br><br>

                    <label for="day">day:</label>
                    <input type="number" id="day" required><br><br>

                    <label for="month">month:</label>
                        <select id="month" required>
                        <option value="jan">jan</option>              
                        <option value="feb">feb</option>              
                        <option value="mar">mar</option>              
                        <option value="apr">apr</option>              
                        <option value="may">may</option>              
                        <option value="jun">jun</option>              
                        <option value="jul">jul</option>              
                        <option value="aug">aug</option>              
                        <option value="sep">sep</option>              
                        <option value="oct">oct</option>              
                        <option value="nov">nov</option>              
                        <option value="dec">dec</option>              
                        </select><br><br>

                    <label for="duration">duration:</label>
                    <input type="number" id="duration" required><br><br>

                    <label for="campaign">campaign:</label>
                    <input type="number" id="campaign" required><br><br>

                    <label for="pdays">pdays:</label>
                    <input type="number" id="pdays" required><br><br>

                    <label for="previous">previous:</label>
                    <input type="number" id="previous" required><br><br>

                    <label for="poutcome">poutcome:</label>
                    <select id="poutcome" required>
                        <option value="unknown">unknown</option>
                        <option value="failure">failure</option>
                        <option value="success">success</option>
                    </select><br><br>

                    <label for="deposit">deposit:</label>
                        <select id="deposit" required>
                        <option value="yes">yes</option>
                        <option value="no">no</option>
                    </select><br><br>
                    <button type="submit">Hesapla</button>
                </form>

                <h2>Sonuç: <span id="sonuc"></span></h2>

                <script>
                    function hesaplamaYap() {
                        var age = parseFloat(document.getElementById("age").value);
                        var job = document.getElementById("job").value;
                        var marital = document.getElementById("marital").value;
                        var education = document.getElementById("education").value;
                        var defaultVal = document.getElementById("default").value;
                        var balance = parseFloat(document.getElementById("balance").value);
                        var housing = document.getElementById("housing").value;
                        var loan = document.getElementById("loan").value;
                        var contact = document.getElementById("contact").value;
                        var day = parseFloat(document.getElementById("day").value);
                        var month = document.getElementById("month").value;
                        var duration = parseFloat(document.getElementById("duration").value);
                        var campaign = parseFloat(document.getElementById("campaign").value);
                        var pdays = parseFloat(document.getElementById("pdays").value);
                        var previous = parseFloat(document.getElementById("previous").value);
                        var poutcome = document.getElementById("poutcome").value;
                        var deposit = document.getElementById("deposit").value;

                        var xhr = new XMLHttpRequest();
                        xhr.open("POST", "/predict", true);
                        xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
                        xhr.onreadystatechange = function () {
                            if (xhr.readyState === 4 && xhr.status === 200) {
                                document.getElementById("sonuc").textContent = "" + xhr.responseText;
                            }
                        };

                        var data = "age=" + age +
                            "&job=" + job +
                            "&marital=" + marital +
                            "&education=" + education +
                            "&default=" + defaultVal +
                            "&balance=" + balance +
                            "&housing=" + housing +
                            "&loan=" + loan +
                            "&contact=" + contact +
                            "&day=" + day +
                            "&month=" + month +
                            "&duration=" + duration +
                            "&campaign=" + campaign +
                            "&pdays=" + pdays +
                            "&previous=" + previous +
                            "&poutcome=" + poutcome +
                            "&deposit=" + deposit;

                        xhr.send(data);
                    }
                </script>

            </body>
            </html>
            '''
            self.wfile.write(content.encode('utf-8'))

    def do_POST(self):
        # Tahmin yapma işlemleri

        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            # Formdan verileri al
            age = float(data['age'][0])
            job = data['job'][0]
            marital = data['marital'][0]
            education = data['education'][0]
            default_val = data['default'][0]
            balance = float(data['balance'][0])
            housing = data['housing'][0]
            loan = data['loan'][0]
            contact = data['contact'][0]
            day = float(data['day'][0])
            month = data['month'][0]
            duration = float(data['duration'][0])
            campaign = float(data['campaign'][0])
            pdays = float(data['pdays'][0])
            previous = float(data['previous'][0])
            poutcome = data['poutcome'][0]
            deposit = data['deposit'][0]

            # Verileri bir DataFrame'e dönüştür
            data_dict = {
                'age': [age],
                'job': [job],
                'marital': [marital],
                'education': [education],
                'default': [default_val],
                'balance': [balance],
                'housing': [housing],
                'loan': [loan],
                'contact': [contact],
                'day': [day],
                'month': [month],
                'duration': [duration],
                'campaign': [campaign],
                'pdays': [pdays],
                'previous': [previous],
                'poutcome': [poutcome],
                'deposit': [deposit]
            }

            df = pd.read_csv("bank.csv")

            new_data = pd.DataFrame(data_dict)
            df = df.append(new_data)

            # Veri ön işleme adımlarını uygula (örneğin, veri dönüşümleri)

            def grab_col_names(dataframe, cat_th=10, car_th=20):
                """

                Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
                Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

                Parameters
                ------
                    dataframe: dataframe
                            Değişken isimleri alınmak istenilen dataframe
                    cat_th: int, optional
                            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
                    car_th: int, optinal
                            kategorik fakat kardinal değişkenler için sınıf eşik değeri

                Returns
                ------
                    cat_cols: list
                            Kategorik değişken listesi
                    num_cols: list
                            Numerik değişken listesi
                    cat_but_car: list
                            Kategorik görünümlü kardinal değişken listesi

                Examples
                ------
                    import seaborn as sns
                    df = sns.load_dataset("iris")
                    print(grab_col_names(df))


                Notes
                ------
                    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
                    num_but_cat cat_cols'un içerisinde.
                    Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

                """

                # cat_cols, cat_but_car
                cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
                num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                               dataframe[col].dtypes != "O"]
                cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                               dataframe[col].dtypes == "O"]
                cat_cols = cat_cols + num_but_cat
                cat_cols = [col for col in cat_cols if col not in cat_but_car]

                # num_cols
                num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
                num_cols = [col for col in num_cols if col not in num_but_cat]

                # print(f"Observations: {dataframe.shape[0]}")
                # print(f"Variables: {dataframe.shape[1]}")
                # print(f'cat_cols: {len(cat_cols)}')
                # print(f'num_cols: {len(num_cols)}')
                # print(f'cat_but_car: {len(cat_but_car)}')
                # print(f'num_but_cat: {len(num_but_cat)}')
                return cat_cols, num_cols, cat_but_car

            def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
                quartile1 = dataframe[col_name].quantile(q1)
                quartile3 = dataframe[col_name].quantile(q3)
                interquantile_range = quartile3 - quartile1
                up_limit = quartile3 + 1.5 * interquantile_range
                low_limit = quartile1 - 1.5 * interquantile_range
                return low_limit, up_limit

            def replace_with_thresholds(dataframe, variable):
                low_limit, up_limit = outlier_thresholds(dataframe, variable)
                dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
                dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

            def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
                dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
                return dataframe

            df = df.drop(labels=["duration", 'poutcome'], axis=1)

            # Drop the Job Occupations that are "Unknown"
            df = df.drop(df.loc[df["job"] == "unknown"].index)
            df = df.drop(df.loc[df["education"] == "unknown"].index)

            # Manager and admin. are basically the same, added under the same categorical value.
            lst = [df]

            for col in lst:
                col.loc[col["job"] == "admin.", "job"] = "management"

            df["NEW_AGE_CAT"] = pd.cut(df['age'], bins=[0, 35, 55, 70, float('Inf')],
                                              labels=['0-35', '35-55', '55-70', '70-100'])

            balance_categories = {
                "debtor": "borçlu",
                'low': 'Düşük',
                'medium': 'Orta',
                'high': 'Yüksek'
            }

            df["NEW_BALANCE_CAT"] = df['balance'].apply(
                lambda x: balance_categories['debtor'] if x < 0 else balance_categories['low'] if x < 1000 else
                balance_categories['high'] if x > 2000 else balance_categories['medium'])

            kış = ["dec", "jan", "feb"]
            ilkbahar = ["mar", "apr", "may"]
            yaz = ["jun", "jul", "aug"]
            sonbahar = ["oct", "nov", "sep"]

            df["NEW_WEATHER_CAT"] = df["month"].apply(
                lambda x: "kış" if x in kış else "ilkbahar" if x in ilkbahar else "yaz" if x in yaz else "sonbahar")

            cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

            cat_cols = [col for col in cat_cols if "deposit" not in col]

            df = one_hot_encoder(df, cat_cols, drop_first=True)

            df.columns = [col.upper() for col in df.columns]

            # Son güncel değişken türlerimi tutuyorum.
            cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
            cat_cols = [col for col in cat_cols if "DEPOSIT" not in col]

            # replace_with_thresholds(df, "BALANCE")
            # replace_with_thresholds(df, "CAMPAIGN")
            # replace_with_thresholds(df, "PDAYS")
            # replace_with_thresholds(df, "PREVIOUS")

            # Standartlaştırma
            X_scaled = StandardScaler().fit_transform(df[num_cols])
            df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

            y = df["DEPOSIT"]
            X = df.drop(["DEPOSIT"], axis=1)







            # Modeli yükle
            new_model = joblib.load("voting_clf1.pkl")

            # Tahmin yap

            user=X.tail(1)

            prediction = new_model.predict(user)

            # Tahmin sonucunu ekrana yazdır
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"{prediction[0]}".encode('utf-8'))






# Sunucu oluşturma ve çalıştırma
if __name__ == '__main__':
    httpd = HTTPServer(server_address, RequestHandler)
    print('Sunucu başlatıldı...')
    httpd.serve_forever()
