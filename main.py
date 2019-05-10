from io import BytesIO
from flask import Flask, render_template, flash, request, send_file, make_response
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug import secure_filename

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.linalg import inv
from haversine import haversine
from sympy import *
import os
import random
import shutil
from datetime import datetime
from itertools import permutations

from matplotlib.figure import Figure                       
from matplotlib.axes import Axes                           
from matplotlib.lines import Line2D                        
from matplotlib.backends.backend_agg import FigureCanvasAgg

global eccData
global costData
global suiData
global ciData
global fiData
global perda
global d
global ext
global extAG
global nomeAG

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/", methods=['GET', 'POST'])
def hello():
	return render_template('index.html')

@app.route("/goToOutdoor", methods=['GET', 'POST'])
def goToOutdoor():
	
	return render_template('fourth.html')

@app.route("/goToIndoor", methods=['GET', 'POST'])
def goToIndoor():
	
	return render_template('fifth.html')

@app.route("/indoor", methods=['GET', 'POST'])
def indoor():
	section = '#rModelos'
	if request.method == 'POST':
		global ext

		now = datetime.now()
		name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)

		try:
			xt = float(request.form['xt'])
			yt = float(request.form['yt'])
			x0 = np.asarray(request.form['x0'].split(",")).astype(np.float)
			y0 = np.asarray(request.form['y0'].split(",")).astype(np.float)
			ptdo = float(request.form['ptd0'])
			do = float(request.form['d0'])
			ptdb = float(request.form['ptdb'])
			f = float(request.form['fq'])
			gt = float(request.form['gt'])
			gr = float(request.form['gr'])
			t = 300
			bmhz = float(request.form['bmhz'])
			noise = 0
			ambiente = request.form['nameAmbiente']

			constb = 1.3806503e-23
			nx = 80
			ny = 40
			Lf = 32.45 + 20 * np.log10(do/1000) + 20 * np.log10(f)
			nap = len(x0)
			cor = 'red'
			dx = np.linspace(0, xt, nx)
			dy = np.linspace(0, yt, ny)
			px = len(dx)
			py = len(dy)
	
			if ambiente[0] == '-':
				file = request.files['myfileIN']
				filename = secure_filename(file.filename) 
				file.save(os.path.join(filename))
				valores, campoeletrico, distancia = lerArquivoIndoor(filename, x0[0], y0[0]) # Lembre de ajeitar isso aqui
				n = calculan(valores, distancia, Lf)
			elif ambiente[0] == 'C':
				n = 1.8
			elif ambiente[0:11] == 'Ambientes G':
				n = 2
			elif ambiente[0:11] == 'Ambientes M':
				n = 3
			elif ambiente[0:11] == 'Ambientes D':
				n = 4
			
			modelo = request.form['nameModels']
			
			if modelo[0] == 'M':
				file1 = request.files['paredes']
				filename1 = secure_filename(file1.filename) 
				file1.save(os.path.join(filename1))
				modelh, modelv, ph, pv = lerTexto(filename1)
			if modelo[0] == 'F':
				file = request.files['myfileIN']
				filename = secure_filename(file.filename) 
				file.save(os.path.join(filename))
				valores, campoeletrico, distancia = lerArquivoIndoor(filename, x0[0], y0[0]) # Lembre de ajeitar isso aqui

		except:
			return render_template('indexError.html')

		if modelo[0] == 'M':
			ext = 'mk'
			tit = 'Motley Keenan'
		elif modelo[0] == 'F':
			ext = 'fi'
			tit = 'Flaoting Interception'
		elif modelo[0] == 'C':
			ext = 'ci'
			tit = 'Close In'
		elif modelo[0] == 'I':
			ext = 'itu'
			tit = 'ITU-R P.1238-8'

		try:
			os.mkdir('static/img/indoor/' + ext)
		except FileExistsError:	
			shutil.rmtree('static/img/indoor/' + ext)
			os.mkdir('static/img/indoor/' + ext)
	
		fig, ax = plt.subplots() 
		if modelo[0] == 'M':
			ax = plotarParedes(ax, ph, pv, modelh, modelv)
			perda_f, non_cob = cobertura(x0, y0, ext, ny, nx, nap, 0, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, 0, 0)
		elif modelo[0] == 'F':
			perda_f, non_cob = cobertura(x0, y0, ext, ny, nx, nap, 0, dx, dy, Lf, n, 0, 0, 0, 0, f, ptdb, gt, gr, py, px, valores, distancia)
		else:
			perda_f, non_cob = cobertura(x0, y0, ext, ny, nx, nap, 0, dx, dy, Lf, n, 0, 0, 0, 0, f, ptdb, gt, gr, py, px, 0, 0)
		perda_f = np.asarray(perda_f) + gt + gr
		ax.plot(x0, y0, 'o', color=cor)
		plt.title("Perda pelo Modelo " + tit)
		plt.imshow(perda_f,cmap='jet',extent=[0,xt,0,yt],origin='lower')
		plt.colorbar(label="dB")

		path = 'static/img/indoor/' + ext + '/perda' + name + '.png'
		plt.savefig(path)

		prmk = ptdb - np.asarray(perda_f)
		fig, ax = plt.subplots()
		if modelo[0] == 'M':
			ax = plotarParedes(ax, ph, pv, modelh, modelv)
		ax.plot(x0, y0, 'o', color=cor)
		plt.title("Potência Recebida pelo " + tit)
		plt.imshow(prmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
		plt.colorbar(label="dBm")

		path = 'static/img/indoor/' + ext + '/pr' + name + '.png'
		plt.savefig(path)

		divisor = np.power(10, noise/10) * constb * t * bmhz * 10**6
		snrmk = np.asarray(prmk) - (10*np.log10(divisor) + 30)
		fig, ax = plt.subplots()
		if modelo[0] == 'M':
			ax = plotarParedes(ax, ph, pv, modelh, modelv)
		ax.plot(x0, y0, 'o', color=cor)
		plt.title("SNR pelo Modelo " + tit)
		plt.imshow(snrmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
		plt.colorbar(label="dB")

		path = 'static/img/indoor/' + ext + '/snr' + name + '.png'
		plt.savefig(path)

		itmk =  np.power(10, ((np.asarray(prmk) + 20 * np.log10(f) + 77.2)/20)) * 0.000001
		fig, ax = plt.subplots()
		if modelo[0] == 'M':
			ax = plotarParedes(ax, ph, pv, modelh, modelv)
		ax.plot(x0, y0, 'o', color=cor)
		plt.title("Intensidade pelo Modelo " + tit)
		plt.imshow(itmk,cmap='jet',extent=[0,xt,0,yt],origin='lower', vmax = 1)
		plt.colorbar(label="V/m")

		path = 'static/img/indoor/' + ext + '/ce' + name + '.png'
		plt.savefig(path)

		ola = (10**((np.asarray(prmk))/10))
		sinrmk = np.asarray(prmk) - (10*np.log10(divisor) + 30) - ola
		fig, ax = plt.subplots()
		if modelo[0] == 'M':
			ax = plotarParedes(ax, ph, pv, modelh, modelv)
		ax.plot(x0, y0, 'o', color=cor)
		plt.title("SINR pelo Modelo " + tit)
		plt.imshow(sinrmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
		plt.colorbar(label="dB")

		path = 'static/img/indoor/' + ext + '/sinr' + name + '.png'
		plt.savefig(path)

		cpmk = []
		for i in range(ny):
			cpmk.append([])
	
		for i in range(ny):
			for j in range(nx):        
				oi = bmhz * np.log2(1+((10**(sinrmk[i][j]/10))/1000))*10**-3
				cpmk[i].append(10*np.log10(1000 * oi))

		fig, ax = plt.subplots()
		if modelo[0] == 'M':
			ax = plotarParedes(ax, ph, pv, modelh, modelv)
		ax.plot(x0, y0, 'o', color=cor)
		plt.title("Capacidade pelo Modelo " + tit)
		plt.imshow(cpmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
		plt.colorbar(label="mbits/s")

		path = 'static/img/indoor/' + ext + '/capacidade' + name + '.png'
		plt.savefig(path)

	return render_template('fifth.html', section=section, ext=ext, name=name)

@app.route("/downIndoor", methods=['GET', 'POST'])
def downloadIndoor():
	if request.method == 'POST':
		try:
			shutil.make_archive('Modelo' + ext, 'zip', 'static/img/indoor/' + ext)
			path =  'Modelo' + ext + '.zip'
		except:
			return render_template('indexError.html')

		return send_file(path, as_attachment=True)
	
	return render_template('fifth.html')

@app.route("/comparacao", methods=['GET', 'POST'])
def comparision():
	algo = 'comparacao'

	if request.method == 'POST':
		#try:
		f = float(request.form['freq1'])
		do = float(request.form['d01'])
		ptdb = float(request.form['ptdb1'])
		gt = float(request.form['gt1'])
		gr = float(request.form['gr1'])
		x0 = float(request.form['x01'])
		y0 = float(request.form['y01'])
		file = request.files['compararFile']
		filename = secure_filename(file.filename) 
		file.save(os.path.join(filename))

		Lf = 32.45 + 20 * np.log10(do/1000) + 20 * np.log10(f)
		valores, campoeletrico, distancia = lerArquivoIndoor(filename, x0, y0)
		pathN, n, Ln5, Ln4, Ln3, Ln2, Ln1, Lnn, dns = calculanComGrafico(valores, distancia, Lf, do)
		pathComparar, itu, ci, mk, o, fi, a, b = comparar(distancia, do, f, Lf, n, ptdb, valores, gt, gr)
		info = str(round(n, 2)) + " " + str(round(rmse(o, itu), 4)) + " " + str(round(rmse(o, ci), 4)) + " " +  str(round(rmse(o, mk), 4)) + " " +  str(round(rmse(o, fi), 4)) + " " + str(round(a, 2)) + " " + str(round(b, 2))
		#except:
		#	return render_template('indexError.html')
	
	return render_template('fifth.html', algo=algo, itu=itu, medido=o, ci=ci, fi=fi, mk=mk, infoCom=info, dist1=distancia, n1=n, Ln5=Ln5, Ln4=Ln4, Ln3=Ln3, Ln2=Ln2, Ln1=Ln1, Lnn=Lnn, dns=dns)

@app.route("/compararDown", methods=['GET', 'POST'])
def comparisionDown():
	algo = 'comparacao'

	if request.method == 'POST':
		try:
			path = 'static/img/indoor/comparacao.png'
			return send_file(path, as_attachment=True)
		except:
			return render_template('indexError.html')
	
	return render_template('fifth.html', algo=algo, itu=itu, medido=o, ci=ci, mk=mk, infoCom=info)

# -----------------------------------------------------------------------------------------------------------------------------------
@app.route("/downSite", methods=['GET', 'POST'])
def downSite():
	path = 'static/pdf/WebZiwiInstrucoes.pdf'
	return send_file(path, as_attachment=True)

@app.route("/downApp", methods=['GET', 'POST'])
def downApp():
	path = 'static/pdf/AplicativoZiwiInstrucoes.pdf'
	return send_file(path, as_attachment=True)

@app.route("/downVR", methods=['GET', 'POST'])
def downVR():
	path = 'static/pdf/VRZiwiInstrucoes.pdf'
	return send_file(path, as_attachment=True)

# -----------------------------------------------------------------------------------------------------------------------------------
@app.route("/indoorAG", methods=['GET', 'POST'])
def oti():
	maisalgo = '#rAG'
	if request.method == 'POST':
		global extAG
		global nomeAG

		now = datetime.now()
		name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
		nomeAG = name

		try:
			tomada = request.form['optradio']
			if tomada[0] == 'c':
				txx = np.asarray(request.form['txx'].split(",")).astype(np.float)
				tyy = np.asarray(request.form['tyy'].split(",")).astype(np.float)
				na = int(request.form['na2'])
				semTomada = ""
				interacoes = 0
				limiar = 0
			else:
				interacoes = int(request.form['intAG'])
				limiar = float(request.form['limAG'])
				semTomada = True

			xt = float(request.form['xt2'])

			yt = float(request.form['yt2'])
			x0 = np.asarray(request.form['x02'].split(",")).astype(np.float)
			y0 = np.asarray(request.form['y02'].split(",")).astype(np.float)
			ptdo = float(request.form['ptd02'])
			do = float(request.form['d02'])
			ptdb = float(request.form['ptdb2'])
			f = float(request.form['fq2'])
			gt = float(request.form['gt2'])
			gr = float(request.form['gr2'])
			t = 300
			bmhz = float(request.form['bmhz2'])
			noise = 0
			ambiente = request.form['nameAmbiente2']
			if ambiente[0] == '-':
				file = request.files['myfileIN2']
				filename = secure_filename(file.filename) 
				file.save(os.path.join(filename))
				valores, campoeletrico, distancia = lerArquivoIndoor(filename, x0[0], y0[0]) # Lembre de ajeitar isso aqui
				n = calculan(valores, distancia, Lf)
			elif ambiente[0] == 'C':
				n = 1.8
			elif ambiente[0:11] == 'Ambientes G':
				n = 2
			elif ambiente[0:11] == 'Ambientes M':
				n = 3
			elif ambiente[0:11] == 'Ambientes D':
				n = 4
			modelo = request.form['nameModels2']
			if modelo[0] == 'M':
				file1 = request.files['paredes2']
				filename1 = secure_filename(file1.filename) 
				file1.save(os.path.join(filename1))
				modelh, modelv, ph, pv = lerTexto(filename1)
		except:
			return render_template('indexError.html')

		constb = 1.3806503e-23
		nx = 80
		ny = 40
		Lf = 32.45 + 20 * np.log10(do/1000) + 20 * np.log10(f)
		nap = len(x0)
		cor = 'red'
		dx = np.linspace(0, xt, nx)
		dy = np.linspace(0, yt, ny)
		px = len(dx)
		py = len(dy)
		
		if modelo[0] == 'M':
			extAG = 'mk'
			tit = 'Motley Keenan'
		elif modelo[0] == 'F':
			extAG = 'fi'
			tit = 'Flaoting Interception'
		elif modelo[0] == 'C':
			extAG = 'ci'
			tit = 'Close In'
		elif modelo[0] == 'I':
			extAG = 'itu'
			tit = 'ITU-R P.1238-8'

		try:
			os.mkdir('static/img/indoor/AG/' + extAG)
		except FileExistsError:	
			shutil.rmtree('static/img/indoor/AG/' + extAG)
			os.mkdir('static/img/indoor/AG/' + extAG)

		# -----------------------------------------------------------------------------------------------------------------------------
		if tomada[0] == 'c':
			if extAG == 'mk':
				ba, by = otimizarTomada(txx, tyy, limiar, na, cor, ph, pv, modelh, modelv, tit, extAG, name, dx, dy, Lf, n, f, ptdb, gt, gr, py, px, ny, nx, xt, yt, 0, 0)
			elif extAG == 'fi':
				ba, by = otimizarTomada(txx, tyy, limiar, na, cor, ph, pv, modelh, modelv, tit, extAG, name, dx, dy, Lf, n, f, ptdb, gt, gr, py, px, ny, nx, xt, yt, valores, distancia)
			else:
				ba, by = otimizarTomada(txx, tyy, limiar, na, cor, 0, 0, 0, 0, tit, extAG, name, dx, dy, Lf, n, f, ptdb, gt, gr, py, px, ny, nx, xt, yt, 0, 0)
			bAP = "Melhor X: " + str(ba) + "\nMelhor Y:  " + str(by)
		else:
			# ---------------------------------------------------------------------------------------------------------------------------
			if extAG == 'mk':
				bestP, bestIndFit, mediumFit = AG(nap, extAG, ny, nx, limiar, interacoes, xt, yt, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, 0, 0)
			elif extAG == 'fi':
				bestP, bestIndFit, mediumFit = AG(nap, extAG, ny, nx, limiar, interacoes, xt, yt, dx, dy, Lf, n, 0, 0, 0, 0, f, ptdb, gt, gr, py, px, valores, distancia)
			else:
				bestP, bestIndFit, mediumFit = AG(nap, extAG, ny, nx, limiar, interacoes, xt, yt, dx, dy, Lf, n, 0, 0, 0, 0, f, ptdb, gt, gr, py, px, 0, 0)

			fig, ax = plt.subplots()
			plt.plot(bestIndFit, label='Fitness do Melhor Indivíduo')
			plt.plot(mediumFit, '-.', label='Fitness Média da População')
			plt.title('Fitness ao Longo das Gerações')
			plt.xlabel('Geração')
			plt.ylabel('Fitness')
			plt.yscale('symlog')
			plt.legend(bbox_to_anchor=(0., 0.9, 1., .1), loc='center', ncol=2, borderaxespad=0)

			path = 'static/img/indoor/AG/' + extAG + '/performance' + name + '.png'
			plt.savefig(path)

			bestX = []
			bestY = []

			for i in range(nap):
				bestX.append(bestP[i * 2])
				bestY.append(bestP[(i * 2) + 1])

			fig, ax = plt.subplots()     

			if modelo[0] == 'M':
				ax = plotarParedes(ax, ph, pv, modelh, modelv) 
				perda_f = cobertura(bestX, bestY, extAG, ny, nx, nap, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, 0, 0)[0]
			elif modelo[0] == 'F':
				perda_f = cobertura(bestX, bestY, extAG, ny, nx, nap, limiar, dx, dy, Lf, n, 0, 0, 0, 0, f, ptdb, gt, gr, py, px, valores, distancia)[0] 
			else:
				perda_f = cobertura(bestX, bestY, extAG, ny, nx, nap, limiar, dx, dy, Lf, n, 0, 0, 0, 0, f, ptdb, gt, gr, py, px, 0, 0)[0]  
			perda_f = np.asarray(perda_f) + gt + gr
			ax.plot(bestX, bestY, 'o', color=cor)
			plt.title("Melhor Perda pelo Modelo Motley Keenan\nMelhor X: " + str(bestX) + "\nMelhor Y: " + str(bestY))
			plt.imshow(perda_f,cmap='jet',extent=[0,xt,0,yt],origin='lower')
			plt.colorbar(label="dB")

			path = 'static/img/indoor/AG/' + extAG + '/perda' + name + '.png'
			plt.savefig(path)

			prmk = ptdb - np.asarray(perda_f)
			fig, ax = plt.subplots()
			if modelo[0] == 'M':
				ax = plotarParedes(ax, ph, pv, modelh, modelv)
			ax.plot(bestX, bestY, 'o', color=cor)
			plt.title("Melhor Potência Recebida pelo " + tit)
			plt.imshow(prmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
			plt.colorbar(label="dBm")

			path = 'static/img/indoor/AG/' + extAG + '/pr' + name + '.png'
			plt.savefig(path)

			divisor = np.power(10, noise/10) * constb * t * bmhz * 10**6
			snrmk = np.asarray(prmk) - (10*np.log10(divisor) + 30)
			fig, ax = plt.subplots()
			if modelo[0] == 'M':
				ax = plotarParedes(ax, ph, pv, modelh, modelv)
			ax.plot(bestX, bestY, 'o', color=cor)
			plt.title("Melhor SNR pelo Modelo " + tit)
			plt.imshow(snrmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
			plt.colorbar(label="dB")

			path = 'static/img/indoor/AG/' + extAG + '/snr' + name + '.png'
			plt.savefig(path)

			itmk =  np.power(10, ((np.asarray(prmk) + 20 * np.log10(f) + 77.2)/20)) * 0.000001
			fig, ax = plt.subplots()
			if modelo[0] == 'M':
				ax = plotarParedes(ax, ph, pv, modelh, modelv)
			ax.plot(bestX, bestY, 'o', color=cor)
			plt.title("Melhor Intensidade pelo Modelo " + tit)
			plt.imshow(itmk,cmap='jet',extent=[0,xt,0,yt],origin='lower', vmax = 1)
			plt.colorbar(label="V/m")

			path = 'static/img/indoor/AG/' + extAG + '/ce' + name + '.png'
			plt.savefig(path)

			ola = (10**((np.asarray(prmk))/10))
			sinrmk = np.asarray(prmk) - (10*np.log10(divisor) + 30) - ola
			fig, ax = plt.subplots()
			if modelo[0] == 'M':
				ax = plotarParedes(ax, ph, pv, modelh, modelv)
			ax.plot(bestX, bestY, 'o', color=cor)
			plt.title("Melhor SINR pelo Modelo " + tit)
			plt.imshow(sinrmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
			plt.colorbar(label="dB")

			path = 'static/img/indoor/AG/' + extAG + '/sinr' + name + '.png'
			plt.savefig(path)

			cpmk = []
			for i in range(ny):
				cpmk.append([])

			for i in range(ny):
				for j in range(nx):        
					oi = bmhz * np.log2(1+((10**(sinrmk[i][j]/10))/1000))*10**-3
					cpmk[i].append(10*np.log10(1000 * oi))

			fig, ax = plt.subplots()
			if modelo[0] == 'M':
				ax = plotarParedes(ax, ph, pv, modelh, modelv)
			ax.plot(bestX, bestY, 'o', color=cor)
			plt.title("Melhor Capacidade pelo Modelo " + tit)
			plt.imshow(cpmk,cmap='jet',extent=[0,xt,0,yt],origin='lower')
			plt.colorbar(label="mbits/s")

			path = 'static/img/indoor/AG/' + extAG + '/capacidade' + name + '.png'
			plt.savefig(path)

			bAP = "Melhor X: " + str(bestX) + "\nMelhor Y:  " + str(bestY)

	return render_template('fifth.html', maisalgo=maisalgo, bap=bAP, nameAG=name, semTomada=semTomada, extAG=extAG)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/otiDown", methods=['GET', 'POST'])
def otidown():
	if request.method == 'POST':
		#try:
		shutil.make_archive('ModeloOtimizado' + extAG, 'zip', 'static/img/indoor/AG/' + extAG)
		path =  'ModeloOtimizado' + extAG + '.zip'

		return send_file(path, as_attachment=True)
		#except:
		#	return render_template('indexError.html')
	
	return render_template('fifth.html')

@app.route("/otiDownTomada", methods=['GET', 'POST'])
def otidowntomada():
	if request.method == 'POST':
		try:
			path = 'static/img/indoor/AG/' + extAG + '/otimizacaoTomada' + nomeAG + '.png'

			return send_file(path, as_attachment=True)
		except:
			return render_template('indexError.html')
	
	return render_template('fifth.html')

@app.route("/choose", methods=['GET', 'POST'])
def chooseEnviroment():
	
	return render_template('second.html')	

@app.route("/cenario", methods=['GET', 'POST'])
def openCenario():
	
	return render_template('third.html')

@app.route("/downAmbiente", methods=['GET', 'POST'])
def downAmbiente():
	path = "static/img/ambiente"
	try:
		os.mkdir(path)
	except FileExistsError:	
		shutil.rmtree(path)
		os.mkdir(path)

	file = open(path + "/ambiente.txt","w") 
	file.write(request.form['texto'])  
	file.close() 
	return send_file(path + '/ambiente.txt', as_attachment=True)

@app.route("/outdoor", methods=['GET', 'POST'])
def hello2():
	if request.method == 'POST':
		global eccData
		global costData
		global suiData
		global ciData
		global fiData
		global perda
		global d
		eccData = []
		costData = []
		suiData = []
		ciData = []
		fiData = []
		perda = []
		d = []

		try:
			f = float(request.form['freq'])
			h = float(request.form['h'])
			lat = float(request.form['lat'])
			longg = float(request.form['long'])
			ptdb = float(request.form['ptdb'])
			#do = float(request.form['d0'])
			gt = float(request.form['gt'])
			gr = float(request.form['gr'])
			file = request.files['myfile']
			ambiente = request.form['nameAmbiente6']
		
			if ambiente[0:8] == 'Cidade G':
				mod = 1
			elif ambiente[0:8] == 'Cidade M':
				mod = 2
			elif ambiente[0:8] == 'Cidade P':
				mod = 3
			filename = secure_filename(file.filename) 
			file.save(os.path.join(filename))

			valores, campoeletrico, distancia = lerArquivo(filename, lat, longg)
			do = min(distancia) * 1000
			Lf = 32.45 + 20 * np.log10(do) + 20 * np.log10(f)
			n = calculan(valores, np.asarray(distancia)/1000, Lf)
		except:
			return render_template('indexError.html')		

		B = valores # potencia recebida
		pontos = distancia # distancia
		D = 10 * np.log10(np.asarray(pontos))
		Num = len(D)
        
		divisor = ((sum(D))**2) - (Num * sum(D**2))
        
		alfaC = (sum(D) * sum(np.asarray(D) * B)) - (sum(D**2) * sum(B)) 
		betaC = (sum(D) * sum(B)) - (Num * sum(np.asarray(D) * B)) 
		alfa = alfaC / divisor
		beta = betaC / divisor

		for i in range(len(distancia)):
			d.append(distancia[i])
			fiData.append(ptdb - (alfa + 10 * beta * np.log10(distancia[i])) - gt - gr)
			eccData.append(ECC(f, h, 1, distancia[i], mod))
			costData.append(Cost231(f, h, 1, distancia[i], mod))
			suiData.append(sui(f, h, 1, distancia[i], mod, distancia))
			ciData.append(closein(Lf, n, distancia[i]))
			perda.append(ptdb - valores [i])

		info = str(round(n, 2)) + " " + str(round(rmse(perda, suiData), 4)) + " " + str(round(rmse(perda, eccData), 4)) + " " +  str(round(rmse(perda, costData), 4)) + " " + str(round(rmse(perda, ciData), 4)) + " " + str(round(rmse(perda, fiData), 4)) + " " + str(round(alfa, 2)) + " " + str(round(beta, 2))

	return render_template('fourth.html', dist=d, val=perda, ecc=eccData, cost=costData, sui=suiData, ci=ciData, fi=fiData, info=info)

@app.route("/down", methods=['GET', 'POST'])
def download():
	if request.method == 'POST':
		try:
			plt.title("Comparação dos Modelos")
			plt.ylabel("Perda - dB")
			plt.xlabel("Distância - km")
			plt.plot(d, perda, ".b", label = "Medidos")
			plt.plot(d, suiData, ".y", label = "SUI")
			plt.plot(d, eccData, ".r", label = "ECC33")
			plt.plot(d, costData, ".g", label = "Cost231")
			plt.plot(d, ciData, ".c", label = "Close IN")
			plt.plot(d, fiData, "xr", label = "Floating Intercpt")
			plt.legend()
			plt.grid(True)
			path = 'static/img/outdoor/outdoor.png'
			plt.savefig(path)
		except:
			return render_template('indexError.html')

		return send_file(path, as_attachment=True)
	
	return render_template('fourth.html', dist=distancia, val=valores)

def calculanComGrafico(valores, distancia, LF, do):
	n = Symbol('n')
	valor = []
	fim = []
	derivada = []
    
	for i in range(len(distancia)):
		valor.append(valores[np.argmin(distancia)] - 10 * n * np.log10(distancia[i]/min(distancia)))
		if valor[-1] != 0:
			fim.append((-1*(valor[i] - valores[i]))**2)
    
	for i in fim:
		derivada.append(diff(i, n))
        
	soma = 0
	for i in derivada:
		soma += i
        
	n = solve(soma)[0]
	#print(soma)
    
	# A partir daki é só pra fazer esse gráfico
    
	d1 = [do, do + do/2, do + do/3]
        
	Ln1 = []
	Ln2 = []
	Ln3 = []
	Ln4 = []
	Ln5 = []
	Lnn = []
	for i in d1:
		Ln1.append(LF + 10 * np.log10(i/do)) 
		Ln2.append(LF + 10 * 2 * np.log10(i/do))
		Ln3.append(LF + 10 * 3 * np.log10(i/do))
		Ln4.append(LF + 10 * 4 * np.log10(i/do))
		Ln5.append(LF + 10 * 5 * np.log10(i/do))
		Lnn.append(LF + 10 * n * np.log10(i/do))

	fig,ax = plt.subplots()
    
	plt.xlabel('Distância - m', size=15)
	plt.ylabel('Perda de Percurso - dB', size=15)

	title = "Valor do Expoente de Atenuação"

	plt.title(title, size=15)

	legenda = "n = " + str(round(n,2))

	plt.semilogx(d1,Ln5,'.k', label="n = 5")
	plt.semilogx(d1,Ln4,'.k', label="n = 4")
	plt.semilogx(d1,Ln3,'.k', label="n = 3")
	plt.semilogx(d1,Ln2,'.k', label="n = 2")
	plt.semilogx(d1,Ln1,'.k', label="n = 1")
	plt.semilogx(d1,Lnn,'or', label=legenda)

	plt.legend(prop={'size':8})

	path = 'static/img/indoor/calculon.png'
	plt.savefig(path)
    
	return path, n, Ln5, Ln4, Ln3, Ln2, Ln1, Lnn, d1

def plotarParedes(ax, ph, pv, modelh, modelv):
    for i in range(len(ph)):
        if modelh[i] == 1:
            l = Line2D(ph[i][0], ph[i][1], color="purple", linewidth=4)                                    
            ax.add_line(l)
        elif modelh[i] == 2:
            l = Line2D(ph[i][0], ph[i][1], color="green", linewidth=3)                                    
            ax.add_line(l)
        elif modelh[i] == 3:
            l = Line2D(ph[i][0], ph[i][1], color="black", linewidth=4)                                    
            ax.add_line(l)
        else:
            print("Valor de parede indefinido")
    
    for i in range(len(pv)):
        if modelv[i] == 1:
            l = Line2D(pv[i][0], pv[i][1], color="purple", linewidth=4)                                    
            ax.add_line(l)
        elif modelv[i] == 2:
            l = Line2D(pv[i][0], pv[i][1], color="green", linewidth=3)                                    
            ax.add_line(l)
        elif modelv[i] == 3:
            l = Line2D(pv[i][0], pv[i][1], color="black", linewidth=4)                                    
            ax.add_line(l)
        else:
            print("Valor de parede indefinido")
    
    return ax

def comparar(distancia, do, f, Lf, n, ptdb, valores, gt, gr):
	itu = []
	fi = []
	mk = []
	ci = []

	B = valores # potencia recebida
	pontos = distancia # distancia
	D = 10 * np.log10(np.asarray(pontos))
	Num = len(D)
        
	divisor = ((sum(D))**2) - (Num * sum(D**2))
        
	alfaC = (sum(D) * sum(np.asarray(D) * B)) - (sum(D**2) * sum(B)) 
	betaC = (sum(D) * sum(B)) - (Num * sum(np.asarray(D) * B)) 
	alfa = alfaC / divisor
	beta = betaC / divisor

	#print(alfa)
	#print(beta)

	dis = np.arange(1, max(distancia) + 1)
	drmse = np.zeros(int(np.ceil(max(distancia))))
	numeros = np.zeros(int(np.ceil(max(distancia))))

	for i in dis:
		#print(i)
		itu.append(20 * np.log10(f) + n * 10 * np.log10(i) - 28) #itu
		fi.append(ptdb - (alfa + 10 * beta * np.log10(i))) 
		mk.append(Lf + 10 * n * np.log10(i/do)) #MK
		ci.append(Lf + 10 * n * np.log10(i)) #ci
        # Floating Interception FI

	for i, v in enumerate(distancia):
		drmse[int(v) - 1] += ptdb - np.asarray(valores[i]) - gt - gr 
		numeros[int(v) - 1] += 1

	for i in range(len(numeros)):
		drmse[i] = drmse[i]/numeros[i]

	o = ptdb - np.asarray(valores) - gt - gr

	#dis = []
	#for d in distancia:
	#	dis.append(d)

	fig,ax = plt.subplots()
    
	plt.xlabel('Distância - m', size=15)
	plt.ylabel('Perda de Percurso - dB', size=15)
	plt.title('Predição da Perda de Percurso', size=15)


	plt.plot(dis, itu, '*b', label='ITU')
	plt.plot(dis, fi, '*k', label='Floating Interception')
	plt.plot(dis, ci, '*y', label='Close In')
	plt.plot(dis, mk, '*c', label='Motley Keenan')
	plt.plot(distancia, o, 'or', label='Dados Medidos')

	plt.legend()

	path = 'static/img/indoor/comparacao.png'
	plt.grid(True)
	plt.savefig(path)
	return path, itu, ci, mk, drmse, fi, alfa, beta

def rmse(medido, predito):
	if len(medido) != len(predito):
		print("vetores tem tamanho diferente")
		return
    
	soma = 0
    
	for i in range(len(medido)):
		soma += np.sqrt(int((medido[i] - predito[i])**2))
    
	return soma / len(medido)

def lerArquivo(arquivo, l1, l2):
	valores = []
	campoeletrico = []
	distancia = []

	file = open(arquivo, "r", encoding="utf8") 
	for line in file:
		if float(line.split(" ")[2]) != 0:
			valores.append(float(line.split(" ")[0]))
			campoeletrico.append(float(line.split(" ")[1]))
			distancia.append(calculadistancia(float(line.split(" ")[2]), float(line.split(" ")[3]), l1, l2))
    
	return valores, campoeletrico, distancia

def lerArquivoIndoor(arquivo, l1, l2):
    valores = []
    campoeletrico = []
    distancia = []

    file = open(arquivo, "r") 
    for line in file:
        valores.append(float(line.split(" ")[0]))
        campoeletrico.append(float(line.split(" ")[1]))
        distancia.append(euclidiana([float(line.split(" ")[2]), float(line.split(" ")[3])], [l1, l2]))
    
    return valores, campoeletrico, distancia

def calculan(valores, distancia, Lf):
    n = Symbol('n')
    valor = []
    fim = []
    derivada = []
    
    for i in range(len(distancia)):
        valor.append(valores[np.argmin(distancia)] - 10 * n * np.log10(distancia[i]/min(distancia)))
        if valor[-1] != 0:
            fim.append((-1*(valor[i] - valores[i]))**2)
    
    for i in fim:
        derivada.append(diff(i, n))
        
    soma = 0
    for i in derivada:
        soma += i
        
    n = solve(soma)[0]
    #print(soma)

    return n

def euclidiana(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    diff = v1 - v2
    quad_dist = np.dot(diff, diff)
    return np.power(quad_dist, 1/2)

def ECC(f, txh, rxh, d, mode):
    if(txh - rxh < 0):
        rxh = rxh/(d/2)
    
    f = f / 1000
    
    gr = 0.789 * rxh - 1.862 # para cidades grandes
    
    afs = 92.4 + 20 * np.log10(d) + 20 * np.log10(f)
    
    abm = 20.41 + 9.83 * np.log10(d) + 7.894 * np.log10(f) + 9.56 * (np.log10(f) * np.log10(f))
    
    gb = np.log10(txh/200) * (13.958 + 5.8 * (np.log10(d) * np.log10(d)))
    
    if(mode > 1): #para cidades medias
        gr = (42.57 + 13.7 * np.log(f)) * (np.log10(rxh) - 0.585)
    
    return afs + abm - gb - gr

def Cost231(f, txh, rxh, d, mode):
	c= 3   # para ambiente urbano
	lrxh = np.log10(11.75 * rxh)
	c_h = 3.2 * (lrxh * lrxh) - 4.97  # cidade grande
	c0 = 69.55
	cf = 26.16
    
	if f > 1500:
		c0 = 46.3
		cf = 33.9
    
	if mode == 2:  # cidade media
		c = 0
		lrxh = np.log10(1.54 * rxh)
		c_h = 8.29 * (lrxh * lrxh) - 1.1
        
	if mode == 3: # cidade pequena
		c = -3
		c_h = (1.1 * np.log10(f) - 0.7) * rxh - (1.56 * np.log10(f)) + 0.8
        
	logf = np.log10(f)
	dbloss = c0 + (cf * logf) - (13.82 * np.log10(txh)) - c_h + (44.9 - 6.55 * np.log10(txh)) * np.log10(d) + c
    
	return dbloss

def cobertura(x, y, modelo, ny, nx, nap, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia):
    perda = []
    perdas = []
    perda_f = []
    
    for i in range(ny):
        perda.append([])
        perda_f.append([])
        
    if modelo == 'fi':
        B = ptdb - valores # perda
        pontos = distancia # distancia
        D = 10 * np.log10(np.asarray(pontos))
        Num = len(D)
        
        divisor = ((sum(D))**2) - (Num * sum(D**2))
        
        alfaC = (sum(D) * sum(np.asarray(D) * B)) - (sum(D**2) * sum(B)) 
        betaC = (sum(D) * sum(B)) - (Num * sum(np.asarray(D) * B)) 
        alfa = alfaC / divisor
        beta = betaC / divisor
    
    for k in range(nap):
        perda = np.zeros((ny,nx));
        for i in range(ny):
            for j in range(nx): 
                d = np.sqrt(np.power(dx[j] - x[k], 2) + np.power(dy[i] - y[k], 2))
                
                if modelo == 'fi':
                    oi = alfa + 10 * beta * np.log10(d)
                    
                if modelo == 'itu':
                    oi = 20 * np.log10(f) + n * 10 * np.log10(d) - 28
                    
                if modelo == 'ci':
                    oi = Lf + 10 * n * np.log10(d) 
                
                if modelo == 'mk':
                    p = 0
                    v = 0
                    pcv = 0
                    freqperdas = [[13, 2, 7], [17, 13, 15], [36, 15, 23]]
            
                    oi = Lf + 10 * n * np.log10(d) 
                
                    a = (dy[i] - y[k]) / (dx[j] - x[k])
                    b = dy[i] - (a*dx[j])
            
                    for w in range(len(ph)):
                        y_test = ph[w][1][0]
                        x_test = (y_test - b)/a
                        x1 = ph[w][0][0]
                        x2 = ph[w][0][1]
                        xr = [x1, x2]
                        yr = [y[k], dy[i]]
            
                        if(y_test < max(yr) and y_test>min(yr) and (x_test<max(xr)) and x_test>min(xr)):
                            if modelh[w] == 1:
                                v += 1
                            elif modelh[w] == 2:
                                pcv += 1
                            elif modelh[w] == 3:
                                p += 1
            
                    for w in range(len(pv)):
                        x_test = pv[w][0][0]
                        y_test = (x_test * a) + b
                        y1 = pv[w][1][0]
                        y2 = pv[w][1][1]
                        yr = [y1, y2]
                        xr = [x[k], dx[j]]
                
                        if(y_test<max(yr) and y_test>min(yr)) and (x_test<max(xr) and x_test>min(xr)):
                            if modelv[w] == 1:
                                v += 1
                            elif modelv[w] == 2:
                                pcv += 1
                            elif modelv[w] == 3:
                                p += 1
            
                    if f <= 1800:
                        perda[i][j] = oi + p * freqperdas[0][0] + v * freqperdas[0][1] + pcv * freqperdas[0][2]
                    elif f <= 3500:
                        perda[i][j] = oi + p * freqperdas[1][0] + v * freqperdas[1][1] + pcv * freqperdas[1][2]
                    else:
                        perda[i][j] = oi + p * freqperdas[2][0] + v * freqperdas[2][1] + pcv * freqperdas[2][2]
                else:
                    perda[i][j] = oi
            
        perdas.append(perda)
    
    for i in range(ny):
        for j in range(nx):
            n_teste = []
            for k in range(nap):
                n_teste.append(perdas[k][i][j])
        
            perda_f[i].append(min(n_teste)) 
            
    cob = ptdb + gt + gr - np.asarray(perda_f) 
    
    cont = 0
    for i in range(py):
        for j in range(px):
            if cob[i][j] < limiar:
                cont += 1
                cob[i][j] = 0
                
    non_cob = (cont/(px * py)) * 100 
    
    return perda_f, non_cob

def _20log10F(x):
    return 8.685889 * np.log10(x)

def sui(f, txh, rxh, d, mode, distancia):
    d *= 1e3 
    
    # Ambiente Urbano
    a = 4.6
    b = 0.0075
    c = 12.6
    s = 8.2
    xhcf = -10.8
    
    # Ambiente Suburbano
    if mode == 2:
        a = 4
        b = 0.0065
        c = 17.1
        xhcf = -10.8
    
    if mode == 3: # rural
        a = 3.6
        b = 0.005
        c = 20
        xhcf = -20
    
    d0 = min(distancia) * 1e3
    A = 20 * np.log10((4 * np.pi * d0)/(300/f))
    y = a - (b * txh) + (c / txh)
    
    # 2.4
    xf = 0
    xh = 0
    
    # maior q 2
    if f > 2000:
        xf = 6 * np.log10(f/2)
        xh = xhcf * np.log10(rxh/2)
        
    return A + (10 * y) * (np.log10(d/d0)) + xf + xh + s

def lerTexto(arquivo):
    ph = []
    pv = []
    modelh = []
    modelv = []
    file = open(str(arquivo), "r", encoding="utf8") 
    for line in file:
        p1 = []
        if line.split(" ")[0] == "h":
            p1 = [[float(line.split(" ")[1]), float(line.split(" ")[2])], [float(line.split(" ")[3]), float(line.split(" ")[4])]]
            ph.append(p1)
            modelh.append(int(line.split(" ")[5]))
        if line.split(" ")[0] == "v":
            p1 = [[float(line.split(" ")[1]), float(line.split(" ")[2])], [float(line.split(" ")[3]), float(line.split(" ")[4])]]
            pv.append(p1)
            modelv.append(int(line.split(" ")[5]))
    
    return modelh, modelv, ph, pv

def startPop(size, n_routers, modelo, ny, nx, limiar, xt, yt, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia):
    pop = []
    for i in range(size):
        oi = []
        for i in range(n_routers):
            oi.append(round(random.uniform(0, xt), 2))
            oi.append(round(random.uniform(0, yt), 2))
        pop.append(oi)

    for i in pop:
        i.append(fitness(i, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia))

    pop.sort(key=lambda x: x[n_routers * 2], reverse=False)
    return pop

def fitness(vec, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia):
    x = []
    y = []
    
    for i in range(n_routers):
        x.append(vec[i * 2])
        y.append(vec[(i * 2) + 1])
        
    non_cob = cobertura(x, y, modelo, ny, nx, n_routers, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia)[1]

    if (non_cob <= 85):
        return non_cob
    # Punição para o caso de o peso exceder o max
    else:
        return non_cob * 2.3

def torneio(pop, tx, nCandidatos, n_routers):
    popPais = []
    
    nFilhos = 2 * int(len(pop) * tx / 2)

    for i in range(nFilhos):
        candidatos = random.sample(pop, nCandidatos)
        
        popPais.append(candidatos[0])
        bestFitness = candidatos[0][n_routers * 2]
        
        for j in candidatos:
            # Quanto menor o non_cob é melhor
            if(j[n_routers * 2] < bestFitness):
                popPais = popPais[:-1]
                popPais.append(j)
                bestFitness = j[n_routers * 2]
    return popPais

def cruzamento(popPais, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia):
    popFilhos = []
    
    for i,k in zip(popPais[0::2], popPais[1::2]):
        filho1 = []
        filho2 = []
        
        mask = np.random.randint(2, size=n_routers * 2)
        
        for j in range(len(mask)):
            if(mask[j]):
                filho1.append(i[j])
                filho2.append(k[j])
            else:
                filho1.append(k[j])
                filho2.append(i[j])
        
        filho1.append(fitness(filho1, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia))
        filho2.append(fitness(filho2, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia))
        popFilhos.append(filho1)
        popFilhos.append(filho2)
    return popFilhos

def inserirFilhos(pop, popFilhos, n_routers):
    pop = pop[:-len(popFilhos)]
    
    for i in popFilhos:
        pop.append(i)
    
    pop.sort(key=lambda x: x[n_routers * 2], reverse=False)
    return pop

def mutacao(pop, tx, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, xt, yt, valores, distancia):
    nMutados = int(len(pop) * tx)
    elitismo = True

    for i in range(nMutados):
        if (elitismo):
            index1 = np.random.randint(1, len(pop) - 1)
        else:
            index1 = np.random.randint(0, len(pop) - 1)
       
        index2 = np.random.randint(0, n_routers * 2)
        
        if index2 % 2 == 0:
            pop[index1][index2] = (round(random.uniform(0, xt), 2))
        else:
            pop[index1][index2] = (round(random.uniform(0, yt), 2))
        
        pop[index1][n_routers * 2] = fitness(pop[index1], n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia)
    pop.sort(key=lambda x: x[n_routers * 2], reverse=False)
    return pop

def AG(n_routers, modelo, ny, nx, limiar, nGeracoes, xt, yt, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia):
    bestIndFit = []
    mediumFit = []

    pop = startPop(100, n_routers, modelo, ny, nx, limiar, xt, yt, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia)

    taxaDeCruzamento = 0.5
    taxaDeMutacao = 0.1
    
    for i in range(nGeracoes):

        popPais = torneio(pop, taxaDeCruzamento, 3, n_routers)
        
        popFilhos = cruzamento(popPais, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia)
        
        pop = inserirFilhos(pop, popFilhos, n_routers)

        pop = mutacao(pop, taxaDeMutacao, n_routers, modelo, ny, nx, limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, xt, yt, valores, distancia)

        #print(f'Geração {i} -- Melhor Indivíduo: {pop[0]} -- non_cob: {pop[0][n_routers * 2]}')

        bestIndFit.append(pop[0][n_routers * 2])
        mediumFit.append(np.mean(pop, axis=0)[n_routers * 2])
    
    return pop[0], bestIndFit, mediumFit

def otimizarTomada(tx, ty, limiar, na, cor, ph, pv, modelh, modelv, tit, extAG, nameAG, dx, dy, Lf, n, f, ptdb, gt, gr, py, px, ny, nx, xt, yt, valores, distancia):
	non = []
	a = list(zip(tx, ty))
	for subset in combinations(a, na):
		tentX = []
		tentY = []
		for i in range(na):
			tentX.append(subset[i][0])
			tentY.append(subset[i][1])
		perda_fi, non_cob = cobertura(tentX, tentY, extAG, ny, nx, len(tentX), limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia)
		non.append([tentX, tentY, non_cob])
    
	fig, ax = plt.subplots() 
	ax.plot(non[0][0], non[0][1], 'o', color=cor)
	if extAG == 'mk':
		ax = plotarParedes(ax, ph, pv, modelh, modelv)
	perda_fi, non_cob = cobertura(non[0][0], non[0][1], extAG, ny, nx, len(tentX), limiar, dx, dy, Lf, n, ph, pv, modelh, modelv, f, ptdb, gt, gr, py, px, valores, distancia)
	plt.title("Perda pelo Modelo " + tit + "\nEixo X: " + str(non[0][0]) + "\nEixo Y: " + str(non[0][1]))
	plt.imshow(perda_fi,cmap='jet',extent=[0,xt,0,yt],origin='lower')
	plt.colorbar(label="dB")

	path = 'static/img/indoor/AG/' + extAG + '/otimizacaoTomada' + nameAG + '.png'
	plt.savefig(path)
        
	non.sort(key=lambda x: x[2], reverse=False)
	return str(non[0][0]), str(non[0][1])

@app.route("/coberturaUnity", methods=['GET', 'POST'])
def coberturaUnity():
	xt = float(request.form['xt'])
	yt = float(request.form['yt'])
	modelo = request.form['modelo']
	x0 = np.asarray(request.form['x0'].split(",")).astype(np.float)
	y0 = np.asarray(request.form['y0'].split(",")).astype(np.float)
	ptdo = float(request.form['ptd0'])
	do = float(request.form['d0'])
	ptdb = float(request.form['ptdb'])
	f = float(request.form['fq'])
	gt = float(request.form['gt'])
	gr = float(request.form['gr'])
	t = 300
	bmhz = float(request.form['bmhz'])
	noise = 0
	n = float(request.form['n'])
	paredess = request.form['paredes']
	
	constb = 1.3806503e-23
	nx = 80
	ny = 40
	Lf = 32.45 + 20 * np.log10(do/1000) + 20 * np.log10(f) + gt + gr
	nap = len(x0)
	dx = np.linspace(0, xt, nx)
	dy = np.linspace(0, yt, ny)
	px = len(dx)
	py = len(dy)

	perda = []
	perdas = []
	perda_f = []
	
	if modelo == 'mk':
		frase = paredess
		ph = []
		pv = []
		modelh = []
		modelv = []
		palavras = frase.split("\n")

		for index, conteudo in enumerate (palavras):
    			p1 = []
    			if(conteudo.split(" ")[0] == "h"):
        			p1 = [[float(conteudo.split(" ")[1]), float(conteudo.split(" ")[2])], [float(conteudo.split(" ")[3]), float(conteudo.split(" ")[4])]]
        			ph.append(p1)
        			modelh.append(int(conteudo.split(" ")[5]))
    			if(conteudo.split(" ")[0] == "v"):
        			p1 = [[float(conteudo.split(" ")[1]), float(conteudo.split(" ")[2])], [float(conteudo.split(" ")[3]), float(conteudo.split(" ")[4])]]
        			pv.append(p1)
        			modelv.append(int(conteudo.split(" ")[5]))
	
	for i in range(ny):
		perda.append([])
		perda_f.append([])
        
	for k in range(nap):
		perda = np.zeros((ny,nx));
		for i in range(ny):
			for j in range(nx): 
				d = np.sqrt(np.power(dx[j] - x0[k], 2) + np.power(dy[i] - y0[k], 2))
                    
				if modelo == 'itu':
					oi = 20 * np.log10(f) + n * 10 * np.log10(d) - 28
                    
				if modelo == 'ci':
					oi = Lf + 10 * n * np.log10(d) 
                
				if modelo == 'mk':
					p = 0
					v = 0
					pcv = 0
					freqperdas = [[13, 2, 7], [17, 13, 15], [36, 15, 23]]
            
					oi = Lf + 10 * n * np.log10(d) 
                
					a = (dy[i] - y0[k]) / (dx[j] - x0[k])
					b = dy[i] - (a*dx[j])
            
					for w in range(len(ph)):
						y_test = ph[w][1][0]
						x_test = (y_test - b)/a
						x1 = ph[w][0][0]
						x2 = ph[w][0][1]
						xr = [x1, x2]
						yr = [y0[k], dy[i]]
            
						if(y_test < max(yr) and y_test>min(yr) and (x_test<max(xr)) and x_test>min(xr)):
							if modelh[w] == 1:
								v += 1
							elif modelh[w] == 2:
								pcv += 1
							elif modelh[w] == 3:
								p += 1
            
					for w in range(len(pv)):
						x_test = pv[w][0][0]
						y_test = (x_test * a) + b
						y1 = pv[w][1][0]
						y2 = pv[w][1][1]
						yr = [y1, y2]
						xr = [x0[k], dx[j]]
                
						if(y_test<max(yr) and y_test>min(yr)) and (x_test<max(xr) and x_test>min(xr)):
							if modelv[w] == 1:
								v += 1
							elif modelv[w] == 2:
								pcv += 1
							elif modelv[w] == 3:
								p += 1
            
					if f <= 1800:
						perda[i][j] = oi + p * freqperdas[0][0] + v * freqperdas[0][1] + pcv * freqperdas[0][2]
					elif f <= 3500:
						perda[i][j] = oi + p * freqperdas[1][0] + v * freqperdas[1][1] + pcv * freqperdas[1][2]
					else:
						perda[i][j] = oi + p * freqperdas[2][0] + v * freqperdas[2][1] + pcv * freqperdas[2][2]
				else:
					perda[i][j] = oi
           
		perdas.append(perda)
   
	for i in range(ny):
		for j in range(nx):
			n_teste = []
			for k in range(nap):
				n_teste.append(perdas[k][i][j])
        
			perda_f[i].append(min(n_teste))

	return str(perda_f)[2:-2] 
	
def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    for indices in permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)

def closein(Lf, n, distancia):
	return Lf + 10 * n * np.log10(distancia)

def calculadistancia(l1, l2, a1, a2):
    return haversine([float(l1), float(l2)], [a1, a2]) # km *1000 eh metro

if __name__ == '__main__': app.run(debug=True)

#if __name__ == '__main__':
#    app.run(host='127.0.0.1', port=8080, debug=True)
