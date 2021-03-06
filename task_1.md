Предисловие
-----------
В каждой задаче необходимо реализовать функцию с заданными входными данными и результатом. Педполагается использование Python3 вкупе с базовыми средствами библиотеки `numpy`, если иное не указано в условии задачи. Шаг интегрирования равен h. Если задач кому-то не хватило, то можно сдавать вдвоём - понимание важнее кода. 

__Примерная структура кода__

```python3
def f(x, y):
	# My code
	return f_value

def your_function(f, x, y):
	# Your code
	return result
```	


Задачи
------

1. Жёсткость

Дано уравнение __y__' =  __f__(x, __y__), где x - число, __y__ - вектор. Определить число жёсткости в заданной точке __y__. Проверить результаты на функции

``` python3
def f(x, y):
	A = np.array([[1,0],[0,x]])
	return A.dot(y)

def rigidity(f, x, y):
	return # blah-blah-blah

x = 5.
y = np.array([2, 3])
print(rigidity(x, y))
# 5
```

Можно использовать функции модуля `numpy.linalg`.


2. Рунге-Кутта (на двух человек)

Дан метод Рунге-Кутты в виде элементов таблицы Бутчера: матрицы __A__ и векторов __b__ и __c__. Метод, возможно, неявный. Проинтегрировать с шагом h c помощью данного метода функцию `y'' + y = 0, y(0)=(1,0), x = [0, 2*pi]`. 

Можно использовать функции модуля `scipy.optimize`, например `scipy.optimize.newton`


3. ОО + ЧН

На отрезке `x = [0, 1]` дано линейное уравнение `y'' + p(x)y' + q(x)y = f(x)` в виде известных функций `p(x), q(x), f(x)` с начальными условиями `y(0) = a, y'(0) = b`. Найти решение методом численного построения общего решения. Для интегрирования можно использовать явный метод Эйлера.


4. Прогонка.

Дана трёхдиагональная матрица __A__ и вектор __b__. Реализовать решение задачи __A__ __y__ = __b__ методом прогонки.


5. Стрельба

Дано уравнение __y__' =  __f__(x, __y__), где x - число, __y__ - вектор длины 2, на отрезке `x = [0, 1]` с граничными условиями y&#8320;(0) = a, y&#8320;(1) = b. Построить метод стрельбы для данной задачи. Найти решение задачи y'' = -arctg(y), y(0) = 0, y(1) = 2. Для интегрирования можно использовать явный метод Эйлера, также можно использовать функции модуля `scipy.optimize`.


6. Квазилинеаризация (со звёздочкой, до трёх человек)

Дано уравнение __y__' =  __f__(x, __y__), где x - число, __y__ - вектор длины 2, на отрезке `x = [0, 1]` с граничными условиями y&#8320;(0) = a, y&#8320;(1) = b. Реализовать метод квазилинеаризации. Недостающие условия додумать. Для решения СЛАУ можно использовать `numpy.linalg.solve`.


7. Ритц

Дана задача (8.5.2 из задачника, где изображён линейный оператор) на отрезке `x = [0, 1]` в виде известных функций `k(x), p(x), f(x)`. Реализовать метод Ритца для решения такой задачи со стандартными базисными функциями. Для решения СЛАУ можно использовать `numpy.linalg.solve`


8. Галёркин

Дана задача (8.6.2 из задачника, где изображён линейный оператор) на отрезке `x = [0, 1]` в виде известных функций `k(x), p(x), f(x)`. Реализовать метод Галёркина для решения такой задачи со стандартными базисными функциями. Для решения СЛАУ можно использовать `numpy.linalg.solve`


9. Адамс

Дано уравнение __y__' =  __f__(x, __y__), где x - число, __y__ - вектор, на отрезке `x = [0, 1]` с начальными условиями __y__(0) = __y0__. Реализовать явный s-стадийный метод Адамса вида `y_(n+1) = a0 * f_n + a1 * f_(n-1) + ... `. Найти решение задачи __y'__ = (0, 0, ..., 1), __y0__ = (0, 0, ..., 0) размерности N.
