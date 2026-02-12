MNIST-projekt för kursen i Machine Learning

Appen är driftsatt och finns för körning här: https://puttaren-predict.streamlit.app/

## 📂 Projektstruktur
–––––––––––––––––––––

### 🏠 Main
* **[predict.py](predict.py)**: Själva Streamlit-applikationen med Live-funktionalitet.
* **[preprocess.py](preprocess.py)**: Den centrala motorn för bildbehandling (ljusanalys, beskärning och tyngdpunkts-centrering).
* **mnist_model_final_svc.joblib**: Den tränade SVC-modellen (98.1% accuracy).
* **requirements.txt**: Alla nödvändiga bibliotek för att köra projektet.

### 📓 [Notebooks/](Notebooks/)
* **MNIST-modellering - final.ipynb**: Slutgiltig genomgång av modeller, hyperparametrar och export av joblib-filen.
* **MNIST-modellering - experiment.ipynb**: EDA och tidiga tester med olika algoritmer (Random Forest, XGBoost m.fl.).
* **Test av preprocessor.ipynb**: Visualisering av hur olika bilder transformeras av preprocessorn.
* **Titta på MNIST-bilder.ipynb**: Utforskning av originaldatasetet.

### 📝 [Teori och självutvärdering/](Teori och självutvärdering/)
* **Teori.txt**: Svar på teorifrågorna (kort och koncist).
* **Självutvärdering.txt**: Mina reflektioner.

### 🎤 [Presentation/](Presentation/)
* **Manus.docx**: Manus för presentationen.

### 📦 [Storage/](Storage/)
* Innehåller gamla modeller, backuper och testbilder.
* *Notera: Den stora Extra Trees-modellen (800MB) finns ej på GitHub p.g.a. storleksgränser.*

## 🛠 Teknik i urval
* **Intelligent Bakgrundsanalys**: Detekterar skuggor i foton och anpassar bildbehandlingen därefter.
* **Tyngdpunkts-centrering**: Flyttar siffrans massa till koordinat 14.0 för att matcha MNIST-standard.
* **SVC (RBF Kernel)**: En optimerad modell som når hög precision på några millisekunder.

Kommentarer:
*   Projektet är fullständigt reproducerbart genom installation av paket enligt 
    requirements.txt, vilken inkluderar alla nödvändiga beroenden för modellering, 
    bildbehandling och visualisering. 

*   Jag frågade om jag behövde gå tillbaka och jobba med dimensionsreducering, men
    fick ju svaret att jag *inte* behövde det så därför är det inte med. Jag har
    experimenterat lite med det och det ger snabbare hantering, men min modell och
    prediktering fungerar ju så det får vara.

*   I den sprudlande experimentlustan tog det lite för lång tid att upptäcka att 
    MNIST-bilderna faktiskt är svarta med vita siffror. När jag kom fram till det 
    och lyckades få bilderna att beskäras rätt och vikta dem så de hamnade rätt i 
    rutan fungerade det *beautifully* i min testmiljö!!!

*   Första idén till en app var att bygga en mailserver på min egen domän som kunde 
    ta emot bilder som bilagor i mail. Det blev dock snabbt en ganska stor uppgift 
    så det får ligga som ett litet frö om jag råkar få massor av tid över. 

