from SimPEG import *
from SimPEG.Utils import sdiag, mkvc
import matplotlib.pyplot as plt

class AcousticTx(Survey.BaseTx):
	

	def __init__(self, loc, time, rxList, **kwargs):

		self.dt = time[1]-time[0]
		self.time = time
		self.loc = loc
		self.rxList = rxList
		self.kwargs = kwargs


	def RickerWavelet(self):

		"""
			Generating Ricker Wavelet

			.. math ::        

		"""
		tlag = self.kwargs['tlag']
		fmain = self.kwargs['fmain']
		time = self.time
		self.wave = (1-2*np.pi**2*fmain**2*(time-tlag)**2)*np.exp(-np.pi**2*fmain**2*(time-tlag)**2)
		return self.wave
	
	def Wave(self, tInd):

		"""
			Generating Ricker Wavelet

			.. math ::        

		"""
		tlag = self.kwargs['tlag']
		fmain = self.kwargs['fmain']
		time = self.time[tInd]
		self.wave = (1-2*np.pi**2*fmain**2*(time-tlag)**2)*np.exp(-np.pi**2*fmain**2*(time-tlag)**2)
		return self.wave

	def getq(self, mesh):

		txind = Utils.closestPoints(mesh, self.loc, gridLoc='CC')
		q = Utils.sdiag(1/mesh.vol)*np.zeros(mesh.nC)
		q[txind] = 1.

		return q

class AcousticRx(Survey.BaseRx):

	def __init__(self, locs, **kwargs):
		self.locs = locs

	# Question: Why this does not work?
	# def getP(self):
	# 	print 'kang'
	# 	return mesh.getInterpolationMat(self.locs, 'CC')
		
	@property
	def nD(self):
		""" The number of data in the receiver."""		
		return self.locs.shape[0]

	def getP(self, mesh):
		P = mesh.getInterpolationMat(self.locs, 'CC')
		return P



class SurveyAcoustic(Survey.BaseSurvey):
	"""
		**SurveyAcousitc**

		Geophysical Acoustic Wave data.

	"""
	
	def __init__(self, txList,**kwargs):
		self.txList = txList
		Survey.BaseSurvey.__init__(self, **kwargs)

	def projectFields(self, u):
		data = []

		for i, tx in enumerate(self.txList):
			Proj = tx.rxList[0].getP(self.prob.mesh)
			data.append(Proj*u[i])
		return data


class AcousticProblem(Problem.BaseProblem):
	""" 

	"""
	surveyPair = SurveyAcoustic
	Solver     = Solver
	storefield = True
	verbose = False
	stability = False
	sig = False

	def __init__(self, mesh, **kwargs):
		Problem.BaseProblem.__init__(self, mesh)
		self.mesh.setCellGradBC('dirichlet')
		Utils.setKwargs(self, **kwargs)

	def setAbsorbingBC(self, npad, const=500.):
		#TODO: performance of abosrbing
		ax = self.mesh.vectorCCx[-npad]
		ay = self.mesh.vectorCCy[-npad]
		indy = np.logical_or(self.mesh.gridCC[:,1]<=-ay, self.mesh.gridCC[:,1]>=ay)
		indx = np.logical_or(self.mesh.gridCC[:,0]<=-ax, self.mesh.gridCC[:,0]>=ax)
		sigx = np.zeros_like(self.mesh.gridCC[:,0])
		sigx[indx] = (abs(self.mesh.gridCC[:,0][indx])-ax)**2
		sigx[indx] = sigx[indx]-sigx[indx].min()
		sigx[indx] = sigx[indx]/sigx[indx].max()
		sigy = np.zeros_like(self.mesh.gridCC[:,1])
		sigy[indy] = (abs(self.mesh.gridCC[:,1][indy])-ay)**2
		sigy[indy] = sigy[indy]-sigy[indy].min()
		sigy[indy] = sigy[indy]/sigy[indy].max()
		L = (self.mesh.hx.sum()-ax)*0.5
		sigmax = L*const
		sig = (sigx+sigy)*sigmax
		sig[indy & indx] = (sig[indy & indx])**1.1*0.5		
		sig[sig>sigmax] = sigmax
		self.sig = sig

	def stabilitycheck(self, v, time, fmain):
		
		self.dxmin = np.min(self.mesh.hx.min(), self.mesh.hy.min())
		self.topt = self.dxmin/v.max()*0.5
		self.dt = time[1]-time[0]
		self.fmain = fmain
		self.wavelen = v.min()/self.fmain
		self.G = self.wavelen/self.dxmin

		if self.dt > self.topt:
			print "dt is greater than topt"
			self.stability = False
		elif self.G < 16.:
			print "Wavelength per cell (G) should be greater than 16"
			self.stability = False
		else:
			self.stability = True


	def fields(self, v):
		
		Grad = self.mesh.cellGrad
		Div = self.mesh.faceDiv
		AvF2CC = self.mesh.aveF2CC		
		rho = 0.27*np.ones(self.mesh.nC)
		mu = rho*v**2


		if self.stability==False:
			raise Exception("Stability condition is not satisfied!!")
		elif self.sig is False:
			print "Warning: Absorbing boundary condition was not set yet!!"

		print ">> Start Computing Acoustic Wave"
		print (">> dt: %5.2e s")%(self.dt)
		print (">> Optimal dt: %5.2e s")%(self.topt)
		print (">> Main frequency, fmain: %5.2e Hz")%(self.fmain)
		print (">> Cell per wavelength (G): %5.2e")%(self.G)
		

		if self.storefield==True:
			P = []			
			#TODO: parallize in terms of sources	
			ntx = len(self.survey.txList)
			for itx, tx in enumerate(self.survey.txList):
				print ("  Tx at (%7.2f, %7.2f): %4i/%4i")%(tx.loc[0], tx.loc[0], itx, ntx)
				pn = np.zeros(self.mesh.nC)
				p0 = np.zeros_like(pn)
				un = np.zeros(self.mesh.nF)
				u0 = np.zeros_like(un)				
				time = tx.time
				dt = tx.dt
				p = np.zeros((self.mesh.nC, time.size))    	    
				q = tx.getq(self.mesh)	
				for i in range(time.size-1):
					sn = tx.Wave(i+1)
					s0 = tx.Wave(i)
					pn = p0-dt*Utils.sdiag(self.sig)*p0+Utils.sdiag(1/rho)*dt*(Div*un+(sn-s0)/dt*q)
					p0 = pn.copy()
					# un = u0 - dt*Utils.sdiag(AvF2CC.T*self.sig)*u0 + dt*Utils.sdiag(1/(AvF2CC.T*(1/mu)))*Grad*p0
					un = u0 + dt*Utils.sdiag(1/(AvF2CC.T*(1/mu)))*Grad*p0
					u0 = un.copy()
					p[:,i+1] = pn

				P.append(p)

			return P

		elif self.storefield==False:

			Data = []

			ntx = len(self.survey.txList)
			for itx, tx in enumerate(self.survey.txList):
				print ("  Tx at (%7.2f, %7.2f): %4i/%4i")%(tx.loc[0], tx.loc[0], itx+1, ntx)
				pn = np.zeros(self.mesh.nC)
				p0 = np.zeros_like(pn)
				un = np.zeros(self.mesh.nF)
				u0 = np.zeros_like(un)
				time = tx.time
				dt = tx.dt				
				data = np.zeros((time.size, tx.nD))
				q = tx.getq(self.mesh)
				Proj = tx.rxList[0].getP(self.mesh)
				for i in range(time.size-1):
					sn = tx.Wave(i+1)
					s0 = tx.Wave(i)					
					pn = p0-dt*Utils.sdiag(sig)*p0+Utils.sdiag(1/rho)*dt*(Div*un+(sn-s0)/dt*q)
					p0 = pn.copy()
					# un = u0 - dt*Utils.sdiag(AvF2CC.T*self.sig)*u0 + dt*Utils.sdiag(1/(AvF2CC.T*(1/mu)))*Grad*p0
					un = u0 + dt*Utils.sdiag(1/(AvF2CC.T*(1/mu)))*Grad*p0
					u0 = un.copy()
					data[i,:] =  Proj*pn		
				
				Data.append(data)

			return Data


if __name__ == '__main__':

	time = np.linspace(0, 0.02, 2**8)
	options={'tlag':0.0025, 'fmain':400}
	rx = AcousticRx(np.vstack((np.r_[0, 1], np.r_[0, 1])))
	tx = AcousticTx(np.r_[0, 1], time, [rx], **options)
	survey = SurveyAcoustic([tx])
	wave = tx.RickerWavelet()
	cs = 0.5
	hx = np.ones(150)*cs
	hy = np.ones(150)*cs
	mesh = Mesh.TensorMesh([hx, hy], 'CC')
	prob = AcousticProblem(mesh)
	prob.pair(survey)
	sig = np.zeros(mesh.nC)
	v = np.ones(mesh.nC)*2000.
	U = prob.fields(v)
	icount = 100
	extent = [mesh.vectorCCx.min(), mesh.vectorCCx.max(), mesh.vectorCCy.min(), mesh.vectorCCy.max()]

	fig, ax = plt.subplots(1, 1, figsize = (5,5))	
	ax.imshow(np.flipud(U[0][:,icount].reshape((mesh.nCx, mesh.nCy), order = 'F').T), cmap = 'binary')
	plt.show()
	
