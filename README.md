# MNIST Digit Classifier – Från Experiment till Feedback-loop

Detta projekt är en djupdykning i bildklassificering med MNIST-datasetet. Resan går från grundläggande teoretiska experiment till ett avancerat jurysystem (Ensemble) och en produktionssatt applikation med inbyggd feedback-funktion för kontinuerlig förbättring.

Appen är driftsatt och finns för körning här: [https://puttaren-predict.streamlit.app/](https://puttaren-predict.streamlit.app/)

## 🧵 Projektets röda tråd
* **1. Teoretisk grund**: Besvarade de teoretiska frågorna kring ML-koncept och Python-objekt för att säkra grundförståelsen.
* **2. Versionshantering**: Etablerade ett arbetsflöde i **GitHub** för att strukturera projektet professionellt (visste att det skulle bli många notebooks).
* **3. Kunskapsinhämtning**: Lärde mig grunderna genom kodexemplet i boken och Scikit-learns dokumentation för att hitta "nyckeln" (Notebook 1).
* **4. Modellsökning**: Jakt på högre accuracy genom att utvärdera olika algoritmer och inställningar (Notebooks 2-4).
* **5. Modellval & Optimering**: Val av slutgiltig huvudmodell och optimering av dess parametrar (Notebook 5).
* **6. Preprocessing & Insikt**: Utveckling av `preprocess.py` baserat på djupanalys av MNIST-datasetets struktur och verifiering via visuella tester.
* **7. Streamlit med feedback-loop**: I appen kan man rita och ladda upp bilder samt ge feedback på felaktiga predikteringar. Dessa bilder sparas ned och kan användas för omträning av modellen.
* **8. Experimentell Accuracy-jakt**: Fortsatta experiment "för sakens skull" med allt möjligt från jurysystem (Ensemble), KNN och SVC-finjusteringar i jakt på mer accuracy (Notebooks 6-14).
* **9. Avslutning**: Finputsade min självutvärdering samt skrev en sammanfattning för presentationen.

## 📓 Notebooks (Experimentlogg)
Det blev många notebooks, men det räcker om du kollar notebook 5–7 där den modell som används i appen skapades. Övriga innehåller mina första stapplande steg inom ML-modellering (notebook 1) och hela vägen upp till en relativt avancerad nivå följt av allmän utforskning.

### 🔍 Analys & Preprocessing
* `Titta på MNIST-bilder.ipynb`: Inledande EDA och visualisering av rådata.
* `Test av preprocessor.ipynb`: Visualisering av hur `preprocess.py` transformerar handritade bilder till maskininläsbart format.

### 🧪 Modelleringsresan (Steg 1-14)

#### Steg 1: Grunden
* `MNIST-modellering 1 - experiment.ipynb`: Första testerna och grundläggande modellval baserat på kursboken.

#### Steg 2-4: Sökandet efter Accuracy
* `MNIST-modellering 2 - jakten på tiondelarna.ipynb`: Finslipning av de inledande modellerna.
* `MNIST-modellering 3 - jakten på tusendelarna.ipynb`: Vidare optimering för att nå maximal precision.
* `MNIST-modellering 4 - utan deskew.ipynb`: Utvärdering av om bild-upprätning (deskewing) faktiskt hjälper resultatet.

#### Steg 5-7: Finalisering & App-val
* `MNIST-modellering 5 - final.ipynb`: Val av modell och export inför app-driftsättning.
* `MNIST-modellering 6 - ett sista försök att maxa accuracy.ipynb`: Slutgiltig push för att nå högsta möjliga poäng.
* `MNIST-modellering 7 - SVC-final.ipynb`: Optimering av den SVC-modell som lade grunden för applikationen.

#### Steg 8-10: Jurysystem (Ensemble)
* `MNIST-modellering 8 - Ensemble.ipynb`: Implementering av jurysystemet (Voting Classifier).
* `MNIST-modellering 9 - KNN.ipynb`: Träning av KNN som ledamot i juryn.
* `MNIST-modellering 10 - Random Forest.ipynb`: Träning av Random Forest som ledamot i juryn.

#### Steg 11-13: Fördjupade tester
* `MNIST-modellering 11 - parameter sweep.ipynb`: Systematisk testning av hyperparametrar.
* `MNIST-modellering 12 - SVC test.ipynb`: Tester av augmentering och särdragsutvinning.
* `MNIST-modellering 13 - SVC no deskew test.ipynb`: Jämförande test för att isolera effekten av preprocessing.

#### Steg 14: Allmän utforskning
* `MNIST-modellering 14 - lek och kladd.ipynb`: Experimentell sandlåda för Hard Negative Mining och vilda idéer.

## 📂 Streamlit-appen
### 🏠 Script och ingående bibliotek
* **predict.py**: Streamlit-appen med rit-/uppladdningsfunktion och feedback-logik.
* **preprocess.py**: "Motorn" som sköter bildbehandlingen av ritade/uppladdade bilder.
* **requirements.txt**: Alla nödvändiga bibliotek för driftsättning.

### 📝 Teoridelen av kunskapskontroll 2
* **Teori.txt**: Svar på teorifrågorna.
* **Självutvärdering.txt**: Utvärderingen och förslag på betyg.

### 📁 Datainsamling & Felanalys
* **mnist_errors/**: Träningsbilder som modellen misslyckades med (används för Hard Negative Mining).
* **collected_data/**: Bilder insamlade via appens feedback-funktion från användare.

## 🛠 Tekniker & Metoder
* **SVC (RBF Kernel)**: Den primära expertmodellen med hög precision.
* **Voting Classifier (Soft Voting)**: Kombinerar sannolikheter från SVC, KNN och RF för stabilitet.
* **Hard Negative Mining**: Strategisk metod för att identifiera och träna på modellens specifika misstag.
* **In-app Feedback Loop**: Användardriven datainsamling för att lösa problem med olika handstilar i realtid.

> **Notera:** Vissa `.joblib`-filer och mappar med stora datamängder exkluderas från GitHub p.g.a. storleksgränser. Kan fås på begäran.
