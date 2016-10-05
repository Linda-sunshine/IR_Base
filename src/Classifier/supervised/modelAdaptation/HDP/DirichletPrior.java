package Classifier.supervised.modelAdaptation.HDP;

import java.util.Arrays;

import cern.jet.random.Gamma;
/**
 * Dirichlet distribution, implemented by gamma function.
 * Referal: https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
 * @author lin
 */
public class DirichletPrior {
	
	//Sampling from parameters [\alpha_1, \alpha_2,...,\alpha_k].
	public void sampling(double[] target, double[] alphas){
		double sum = 0;
		for(int i=0; i<alphas.length; i++){
			while(target[i] == 0)
				target[i]= Gamma.staticNextDouble(alphas[i], 1);
			sum += target[i];
		}
		for(int i=0; i<target.length; i++) 
			target[i]/=sum;
	}
	//Sampling each dim of given target vector.
	public void sampling(double[] target, double alpha){
		double[] alphas = new double[target.length];
		Arrays.fill(alphas, alpha/target.length);
		sampling(target, alphas);
	}
	
	//Sampling given dim of target vector.
	public void sampling(double[] target, int dim, double alpha){
		double[] alphas = new double[dim];
		Arrays.fill(alphas, alpha/dim);
		sampling(target, alphas);
	}
}
