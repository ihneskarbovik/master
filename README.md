# master



## TODO
* I test11 og test12 må jeg ende 0 og 1 på y-aksen til 4g og 5g
* Kjøre alt på nytt og lagre med resultater (i hvertfall sinr_pred, rsrp_pred og rsrq_pred)


## Notater fra møter
* Prøv med enda ferre steg bakover med LSTM (Kostas foreslo å vise dette i en tabell)

* Hvis det er overlappende ruter på OP1 og OP2 kan jeg prøve å trene på en OP og teste på den andre OP

* Test med samme ruter og tren med en og test med en annen
    - Må gjøres systematisk; tren 1, test 1 --> tren 2, test 1 --> tren 3, test 1 --> tren 4, test 1 --> tren 5, test 1
    - Kan også vurdere (men teori er at det funker dårligere): test på 70% av to ruter (overlappende 70%) og test på gjenværende 30% av begge rutene, med samme strategi som over

* Hvis jeg tester med samme rute, kan vi prøve med lokasjon!! -- var ikke så mye å hente på det, ref similar_campaigns.ipynb

* Hvis vi trener på en rute og tester på en annen, funker det dårlig? Hvis testing på samme rute funker bra?


* Prøv å isolere features

* Se på accuracy på 1. steg, 1. + 2. steg, 1. + 2. + 3. steg, osv. Se om accuracy dropper og hvor stort vinduet men fortsatt ha en gitt sikkerhet på at det kommer til å inntreffe en HO