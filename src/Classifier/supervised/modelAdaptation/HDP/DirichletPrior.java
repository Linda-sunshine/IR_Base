package Classifier.supervised.modelAdaptation.HDP;

import java.util.Arrays;

import cern.jet.random.tdouble.Gamma;
/**
 * Dirichlet distribution, implemented by gamma function.
 * Referal: https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
 * @author lin
 */
public class DirichletPrior {
	
	//Sampling from parameters [\alpha_1, \alpha_2,...,\alpha_k].
	public void sampling(double[] target, double[] alphas, boolean toLog){
		double sum = 0;
		for(int i=0; i<alphas.length; i++){
			while(target[i] == 0)
				target[i] = Gamma.staticNextDouble(alphas[i], 1);
			sum += target[i];
		}
		
		for(int i=0; i<alphas.length; i++) {
				target[i]/=sum;
			if (toLog)
				target[i] = Math.log(target[i]);
		}
	}
	
//	//Sampling each dim of given target vector.
//	public void sampling(double[] target, double alpha){
//		double[] alphas = new double[target.length];
//		Arrays.fill(alphas, alpha/target.length);
//		sampling(target, alphas);
//	}
	
	//Sampling given dim of target vector.
	public void sampling(double[] target, int dim, double alpha, boolean toLog){
		if (dim>target.length) {
			System.err.println("[Error]The length of target vector is shorter than the specified size!");
			return ;
		}
		
		double[] alphas = new double[dim];
		Arrays.fill(alphas, alpha);
		sampling(target, alphas, toLog);
	}
}
