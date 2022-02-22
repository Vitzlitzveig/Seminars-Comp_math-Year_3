#!/usr/bin/env python3

import pyhsfcgns as ph
import numpy as np
import scipy as scp
from scipy.optimize import curve_fit, newton
import matplotlib.pyplot as plt

eps = 1e-8


class RgK:

	def __init__(self, A, b, c):
		self.A = A
		self.b = b
		self.c = c
		self.explicit = self.is_explicit()


	# Print Butcher table
	def __str__(self):
		result = ''
		for ai, ci in zip(A, c):
			result += f'{float(ci):7.5} | ' + \
			          ''.join([f'{float(a):7.5} ' for a in ai]) + '\n'

		result += '_'*8 + '|' + '_'*(7+1)*len(ai) + '\n' + \
		          f'{"":7} | ' + ''.join([f'{float(bi):7.5} ' for bi in b]) + '\n'

		return result


	# Calculate one step
	def __call__(self, f, x, y, h):
		if self.explicit:
			return self.calc_explicit(f, x, y, h)
		else:
			return self.calc_implicit(f, x, y, h)


	# Check Butcher table if it refer to an explicit RgK method
	def is_explicit(self):
		for i in range(len(A)):
			if any(A[i, i:]):
				return False
		return True


	# Calculate explicit RgK step
	def calc_explicit(self, f, x, y, h):
		s = len(b)
		if y.__class__ == np.ndarray:
			ff = np.zeros( (s, len(y)) )
		else:
			ff = np.zeros((s,))

		for i, (ci, Ai) in enumerate(zip(self.c, self.A)):
			ff[i] = f(x + ci*h, y + h*Ai.dot(ff))
		return y + h * self.b.dot(ff)


	# Calculate implicit RgK step
	def calc_implicit(self, f, x, y, h):
		def RgK_fun(ff, f, x, y, h):
			z = zip(self.c, self.A)
			return ff - np.array([ f(x + h*ci, y + h*ff.dot(Ai)) for ci, Ai in z ])

		ff0 = np.ones_like(self.b) * f(x, y)
		ff = newton(RgK_fun, ff0, args=(f, x, y, h))
		return y + h*ff.dot(self.b)


	# Get experimental order of RgK method
	def get_ord(self):
		y0 = 1.
		x0 = 0.
		f = lambda x, y: y
		F = lambda x: np.exp(x0 + x)
		hh = np.array([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3])

		if self.is_explicit():
			yy = [self.calc_explicit(f, x0, y0, h) for h in hh]
		else:
			yy = [self.calc_implicit(f, x0, y0, h) for h in hh]

		EE = np.abs( yy - F(x0 + hh) )
		lgH, lgE = np.log(hh), np.log(EE)
		lin = lambda lgh, a, pow: a + pow*lgh
		a, ord = curve_fit(lin, lgH, lgE, (0, 1))[0]
		# print(EE)
		# plt.loglog(hh, EE, '-o')
		# plt.loglog(hh, np.exp(a)*hh**ord)
		# plt.show()
		return round(ord - 1)


	# Stability function R(z) for RgK method
	def stab_func(self, z):
		try:
			one = np.ones_like(self.b)
			E = np.diag(one)
			l_EzA = np.linalg.inv(E - z*self.A)
			return 1 + z * b.dot(l_EzA).dot(one)
		except np.linalg.LinAlgError:
			return np.nan

	def get_stab_func(self):
		return np.vectorize(self.stab_func)


# Print Butcher's table and test method's order
def test_method(rgk):
	print('##### Butcher\'s table #####')
	print(rgk)
	print()

	npR = rgk.get_stab_func()

	xx = np.linspace(-5, 5, 100)
	yy = np.linspace(-5, 5, 100)
	XX, YY = np.meshgrid(xx, yy)
	ZZ = XX + 1j*YY
	aRR = np.abs(npR(ZZ))
	print(f'lim R(z) = {R(-1e10):0.5}\nz -> -inf')
	print(f'Order = {rgk.get_ord()}')

	levels=np.linspace(0, 3, 101)
	plt.contourf(XX, YY, aRR, levels=levels, extend='max', cmap='gray')
	plt.colorbar(ticks=(0, 1, 2, 3))
	plt.contour(XX, YY, aRR, levels=[1], colors='k')
	# plt.contourf(XX, YY, aRR, levels=[0, 1], colors='k', alpha=0.3)
	plt.scatter(XX, YY, aRR <= 1.0, c='k', alpha=0.3)
	plt.title('$|R(z)|$ и область устойчивости. Метод Рунге-Кутты.')
	plt.xlabel(r'$Re(z)$')
	plt.xlim(xx[0], xx[-1])
	plt.ylabel(r'$Im(z)$')
	plt.ylim(yy[0], yy[-1])
	plt.grid()
	plt.tight_layout()
	plt.savefig('RK_example2.png', dpi=600)
	plt.show()


# Single shooting method test
def shot(rgk, f, xx, y0):
	yy = np.zeros((xx.shape[0], 2))
	yy[0] = y0
	for n, (xn, yn) in enumerate(zip(xx[:-1], yy[:-1])):
		yy[n+1] = rgk(f, xn, yn, h)
	return yy


# Shooting method
def shooting(rgk, f, xx, y0, y1, dy0_dx0):
	shot_fun = lambda dy0_dx: y1 - shot(rgk, f, xx, (y0, dy0_dx))[-1, 0]
	dy_dx = newton(shot_fun, dy0_dx0)
	return dy_dx, shot(rgk, f, xx, (y0, dy_dx))


if __name__ == '__main__':
	A = np.array([[0, 0],
	              [1, 0]])
	c = np.array([0, 1])
	b = np.array([1/2, 1/2])

	rgk = RgK(A, b, c)
	# test_method(rgk)

	f = lambda x, y: np.array((y[1], -y[0]**3))
	xx = np.linspace(0, 1, 1001)
	y0 = 0.
	y1 = 2.
	h = xx[1] - xx[0]

	prev = np.nan
	lines = ['.-', ':', '-.', '--', '-']
	for dy0_dx0 in np.linspace(0, 200, 5):
		dy_dx, yy = shooting(rgk, f, xx, y0, y1, dy0_dx0)
		if np.abs(dy_dx - prev) < 1e-6:
			continue
		plt.plot(xx, yy[:, 0], lines.pop(), c='k', label=rf'${{\delta y}}/{{\delta x}} = {dy_dx:5.5}$')
		prev = dy_dx

	plt.legend()
	plt.grid()
	plt.title(f'Решения уравнения $y\'\'-y^3=0, y_0={y0}, y_1={y1}$')
	plt.savefig('shooting method.png', dpi=600)
	plt.show()
