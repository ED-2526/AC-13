21/11/25: Projecte Twitter. Hacer para el próximo día (un punto de la nota final del proyecto):
•	venir con el GitHub hecho el dataset cargado (recomiendan coger más datos de los que dan ellos).
•	Analizar datos que variables hay, si es multi class, si está esbiaixat, ser capaces de cargar el dataset y seleccionar los diferentes conjuntos de test, train, validation, etc.

•	Manipularlo y entender la estructura de datos.
Estructura de les dades:
-	1.600.000 tweets, cada un és positiu “4” neutral “2, o negatiu “0”

-	Camps:
	target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
	ids: The id of the tweet ( 2087)
	date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
	flag: The query (lyx). If there is no query, then this value is NO_QUERY.
	user: the user that tweeted (robotickilldozr)
	text: the text of the tweet (Lyx is cool)

-	Teoricament hauriem d’estar treballant amb un dataset multiclass, però només hi ha 2 classes, 0 i 4, per tant és binari i com està equilibrat al 50% en aquestes dues classes és doncs, equilibrat.
  ![Diagrama de flujo del sistema](provisional/imagen1.png)
 	![Diagrama de flujo del sistema](provisional/imagen2.png)
-	Hem fet una comprovació i resulta que hi ha twits duplicats, haurem de netejar les dades. Teòricament hi ha 79,390 twits buits o duplicats

•	Setup para el Cross validation.
![Diagrama de flujo del sistema](provisional/imagen3.png)
•	Y un starting point del proyecto.
•	Tener controlados los datos. Si no tenemos claro que hay que hacer, contactar un tiempo antes al profesor que nos lleve el proyecto. Quieren que probemos varios códigos para comparar métodos.

