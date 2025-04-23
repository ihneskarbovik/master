# master

## TODO
* I test11 og test12 må jeg ende 0 og 1 på y-aksen til 4g og 5g
* Kjøre alt på nytt og lagre med resultater

* Prøve å vekte klassene til RF!!!


## Notater fra møter
* 👊🏼 Prøv med enda ferre steg bakover med LSTM (Kostas foreslo å vise dette i en tabell)

* Hvis det er overlappende ruter på OP1 og OP2 kan jeg prøve å trene på en OP og teste på den andre OP -- og motsatt

* 👊🏼 Test med samme ruter og tren med en og test med en annen
    - 👊🏼 Må gjøres systematisk; tren 1, test 1 --> tren 2, test 1 --> tren 3, test 1 --> tren 4, test 1 --> tren 5, test 1
    - 👊🏼 Kan også vurdere (men teori er at det funker dårligere): test på 70% av to ruter (overlappende 70%) og test på gjenværende 30% av begge rutene, med samme strategi som over

* 👊🏼 Hvis jeg tester med samme rute, kan vi prøve med lokasjon!! -- var ikke så mye å hente på det, ref similar_campaigns.ipynb

* 👊🏼 Hvis vi trener på en rute og tester på en annen, funker det dårlig? Hvis testing på samme rute funker bra?


* Se på accuracy på 1. steg, 1. + 2. steg, 1. + 2. + 3. steg, osv. Se om accuracy dropper og hvor stort vinduet men fortsatt ha en gitt sikkerhet på at det kommer til å inntreffe en HO
    - 👊🏼 Gjøre det samme med mean absolute error

* 👊🏼 Se om jeg kan omgjøre gps dataen til en kolonne i stedet for lat og long

* Må legge RF oppå LSTM, teste med det vinduet og se på accuracy som beskrevet tidligere!

* 👊🏼 Marker rett og gale prediksjoner med farge og scatterplot for å se om det er et sånn 'range of theshold': plot med yakse: sss-rsrp, x-akse:rsrp maybe?? experiment with parameters on the x and y axis
    - Burde starte med å plotte bare 4G, 5G på samme måte
    - Er det buffer periode mellom ho? for å unngå pingpong effekt (hysterizis). Range of theshold?




Notat: ss og vanlig er inne 1:1 comparable men begge er logaritmic
