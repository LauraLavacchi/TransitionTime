

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>  // for setting precision, see http://stackoverflow.com/questions/5907031/printing-the-correct-number-of-decimal-points-with-cout
#include <math.h>       /* round, floor, ceil, trunc */ 
#include <stdlib.h>     /* atof */
#include <stdio.h> /* for printf */
#include <cmath>       /* sqrt */ 

#include <random>

 // Split string at delimiter
 // Source: http://code.runnable.com/VHb0hWMZp-ws1gAr/splitting-a-string-into-a-vector-for-c%2B%2B
#include <vector>

// Load parameters
#include "parameters.h" 
const int N_dim = 3 ;



double F_det(double position)
{
		return -4*V0*(position*position-1)*position ;
}


std::vector <double> F_d(std::vector <double> x, double RandomForce)
{
	// calculate deterministic force
	double F_deterministic ;
	F_deterministic = F_det(x[0]) ;

	std::vector <double> result (N_dim,0.) ;
	result[0] = x[1] ;
	result[1] = td_by_tm*(x[2]  + F_deterministic) ; 
	result[2] = -td_by_tg*(x[2] + x[1] - RandomForce) ;
	return result ;
}

 
int main( int argc , char** argv )
{
	std::vector <double> x (N_dim,0.) ;
	x[0] = x0 ; // set initial position
	
// histogram
	std::vector <int> Counts (histBins) ;
	std::ofstream HistogramOut;

	//x[0] = 3 ;

	std::ofstream TrajOut;
  	TrajOut.open ("trajectory.txt");

	// set output precision, see 
	// http://stackoverflow.com/questions/5907031/printing-the-correct-number-of-decimal-points-with-cout
    std::cout << std::setprecision(13) ;
    
    /* determine timestep
     We want that
     \Delta t << 1, 
     t_{D}/t_{m} \Delta t << 1 and
     t_{D}/t_{\Gamma} \Delta t << 1
    */gh 
    if (td_by_tm*dt > 0.01)
    {
    	dt = 0.01*tm_by_td ;
    }
    if (td_by_tg*dt > 0.01)
    {
    	dt = 0.01*tg_by_td ;
    }
    
// cbjRandomPrefac = std::sqrt(4*td_by_tg/dt) ;
    RandomStdDev = std::sqrt(2./dt) ;  
	std::random_device rd;
    std::mt19937 gen(rd());
	std::normal_distribution<> Gauss(0,RandomStdDev);


	const long N_steps = long(T_sim/dt) ;
	const long N_out = long(dt_out/dt) ; 
	const long NumberOfStepsToDiscardAtBeginning = long(tg_by_td/dt) ;
	// std::cout << "N_steps = " << N_steps << std::endl;


	std::cout << "dt = " << dt << "\nN_steps = " << N_steps << "\nN_out = " << N_out << std::endl;
	std::cout << "NumberOfStepsToDiscardAtBeginning = " << NumberOfStepsToDiscardAtBeginning << std::endl;




	for (long int n = -NumberOfStepsToDiscardAtBeginning; n < N_steps+1; n++)
	{
		std::vector <double> x_temp (N_dim,0.) ;
		std::vector <double> K1 (N_dim,0.) ;
		std::vector <double> K2 (N_dim,0.) ;
		std::vector <double> K3 (N_dim,0.) ;
		std::vector <double> K4 (N_dim,0.) ;
		
		// random force
		double F_r = Gauss(gen) ;

		// K1
		K1 = F_d(x,F_r) ;
		// K2
		for (int i = 0; i < N_dim ; i++)
		{
			x_temp[i] = x[i] + dt*K1[i]/2. ;
		}
		
		K2 = F_d(x_temp,F_r) ; 
		// K3
		for (int i = 0; i < N_dim ; i++)
		{
			x_temp[i] = x[i] + dt*K2[i]/2. ;
		}
		
		K3 = F_d(x_temp,F_r) ; 
		// K4
		for (int i = 0; i < N_dim ; i++)
		{
			x_temp[i] = x[i] + dt*K3[i] ;
		}
		
		K4 = F_d(x_temp,F_r) ; 
		
		// new vector
		for (int i = 0; i < N_dim; i++)
		{
			x[i] = x[i] + dt*(K1[i] + 2*K2[i] + 2*K3[i] + K4[i])/6. ;
		}


		// histogram 
		double CurPos= x[0]-xLeft ; // shift from (xLeft,xRight) to (0,xLeft+xRight)
		// add value to histogram
		int CurIndex = (int)(CurPos/histDelta) ;
		// std::cout << CurIndex << std::endl ;
		Counts[CurIndex]++ ;
		/*if (0 <= n) 
		{
			if (n <= 10)
			{
				std::cout << CurPos << "  " << CurIndex << std::endl ;
			}
		}
		*/
		
		// output
		if (n >= 0)
		{
			/*if (n % 1000 == 0)
			{
				std::cout << "progress: " << double(n)/double(N_steps)*100. << " %\r" ; 
			}*/
			if (n % N_out == 0)
			{
				TrajOut << n*dt << "\t" << x[0] ;
				if ( V_out )
				{
					TrajOut << "\t" << x[1]  ;
				}
				TrajOut << "\n" ;
				//std::cout << "t = " << n*dt/1000 << " ns (of " << T_sim/1000 << " ns)               \r" ;
			}
		}
	}
	std::cout << std::endl;
	
  	HistogramOut.open ("Position_Histogram.txt");
	HistogramOut.precision(14) ;
	HistogramOut << "# number of datapoints = " << N_steps << std::endl ;
	HistogramOut << "# histMin     = " << histMin << " degrees" << std::endl;
	HistogramOut << "# histMax     = " << histMax << " degrees" << std::endl;
	HistogramOut << "# histDelta   = " << histDelta << " degrees" << std::endl;
	HistogramOut << "# column    content" << std::endl;
	HistogramOut << "#    0      Middle of interval [i*dx+histMin,(i+1)*dx+histMin] (0 <= i <= (histMax-histMin)//dx)" << std::endl;
	HistogramOut << "#    1      counts in that interval" << std::endl;
	for(int i=0; i<histBins; ++i)
	{
		HistogramOut << (i+0.5)*histDelta+histMin << "\t" << Counts[i] << std::endl ;
	}	
  	HistogramOut.close();


 	return 0 ;
}
