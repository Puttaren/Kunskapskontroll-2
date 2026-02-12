Filer för kunskapskontroll 2.
–––––––––––––––––––––––––––––

** Filerna kan med fördel läsas/hanteras i följande ordning:
1. Teori.txt - svar på teorifrågorna. Kort och koncist som efterfrågats. 
2. MNIST-modellering - experiment.ipynb - EDA, allmän visualisering av data, diverse experiment och tester.
3. Titta på MNIST-bilder.ipynb - en tjuvtitt på bilder från MNIST-datasetet. Mycket upplysande!
4. MNIST-modellering - final - Här finns genomgången av olika modeller samt hyperparametrar och skapande av joblib-modellen.
5. preprocess.py - bildbehandling av egna bilder som ska predikteras.
6. Test av preprocessor.ipynb - experimentering med olika bilder för att se hanteringen.
7. ??? <Streamlit-appen> 

Kommentarer:
* Projektet är fullständigt reproducerbart genom den bifogade requirements.txt, vilken inkluderar alla nödvändiga beroenden för både modellering, bildbehandling (Pillow/Scipy) och visualisering. 

* I den sprudlande experimentlustan tog det lite för lång tid att upptäcka att MNIST-bilderna faktiskt är svarta med vita siffror. När jag kom fram till det och lyckades få bilderna att beskäras rätt och vikta dem så de hamnade rätt i rutan fungerade det *beautifully*!!!

