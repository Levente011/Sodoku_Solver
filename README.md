# Sudoku Solver

A program képes egy kamera elé tartott sodoku rejtvényt felismerni, majd megoldani azt. A felismerést gépi látással valósítja meg, majd amennyiben megoldható a feladat a megoldott rejtvényt egy `solved.png` fájlban elmenti a gyökérkönyvtárban és megjeleníti azt a felhasználó számára.

## Telepítés és futtatás

1. A futtatáshoz szükséges könyvtárakat a `requirements.txt` fájl tartalmazza. Ezek a következő paranccsal telepíthetők:

   ```bash
   pip install -r requirements.txt

2. A forráskódban található konstansokat a `config.txt` fájl módosításával lehet áttírni. Ez a fájl tartalmazza a tanításhoz szükséges képek elérési útvonalát, valamint a kamera képének forrását.

3. (Opcionális) A program egy androidos telefonra telepített [IP Camera](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en&pli=1) használatával lett tesztelve, így ennek a telepítése ajánlott, de nem szükséges.

4. A program a következő paranccsal futtatható a gyökérkönyvtárból:

     ```bash
   python main.py

## Meoldott sodoku

<p align="center">
  <img src="https://github.com/Levente011/Sodoku_Solver/blob/main/solved.png?raw=true" width="500" height="500" alt="Megoldott Sudoku">
</p>

