package topicModel;

import java.util.Random;

public abstract class TopicModel {
	
	protected int vocabulary_size;
	protected int number_of_iteration;
	protected double lambda;
	
	/*p (w|theta_b) */
	protected double background_probability [];
	
	public abstract void calculate_E_step();
	public abstract void calculate_M_step();
	public abstract double calculate_log_likelihood();
	public abstract void initialize_probability();
	
	
	
	public void EM()
	{
		
		calculate_E_step();
		calculate_M_step();
		double initial = calculate_log_likelihood();
		double delta = 100000000.0;
		int  i = 0;
		while (delta>.000000001 && i<this.number_of_iteration)
		{
			System.out.println("Likelihood at :" + i + " "+ initial);
			calculate_E_step();
			calculate_M_step();
			double later = calculate_log_likelihood();
			delta = (Math.abs(Math.abs(initial) -  Math.abs(later)))/Math.abs(initial);
			initial = later;
			i++;
			
		}
	}
	
	
	public double[] randomProbilities(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("The size param must be greate than zero");
        }
        double[] pros = new double[size];

        int total = 0;
        Random r = new Random();
        for (int i = 0; i < pros.length; i++) {
            //avoid zero
            pros[i] = r.nextInt(size) + 1;

            total += pros[i];
        }

        //normalize
        for (int i = 0; i < pros.length; i++) {
            pros[i] = (double )pros[i] / total;
        }

        return pros;
    }
	
}
