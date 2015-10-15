/**
 * 
 */
package mains;

import java.awt.List;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Map;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Locale.Category;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.Set;

public class accumulatorKappa {
	
	double [] normal;
	double [] intra;
	double [] inter;
	
	
    double getMean(double [] data)
    {
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/data.length;
    }

    double getVariance(double [] data)
    {
        double mean = getMean(data);
        double temp = 0;
        for(double a :data)
            temp += (mean-a)*(mean-a);
        return temp/data.length;
    }

    double getStdDev(double [] data)
    {
        return Math.sqrt(getVariance(data));
    }
	
	
	public void readcsv(String fileName, String category)
	{
		if(category.equalsIgnoreCase("tv"))
		{
			normal = new double [3];
			intra = new double [3];
			inter = new double [3];
		}
		else{
			normal = new double [6];
			intra = new double [6];
			inter = new double [6];
		}
			
		
		BufferedReader fileReader = null;
		try {
	 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			int normalCounter = 0;
			int intraCounter = 0;
			int interCounter = 0;
			while ((line = fileReader.readLine()) != null) {
				if(line.contains("Normal")){
					normal[normalCounter] = Double.parseDouble(line.split(",")[1]);
					normalCounter++;
				}
				
				if(line.contains("Intra")){
					intra[intraCounter] = Double.parseDouble(line.split(",")[1]);
					intraCounter++;
				}
				
				if(line.contains("Inter")){
					inter[interCounter] = Double.parseDouble(line.split(",")[1]);
					interCounter++;
				}
			}
			
			/*System.out.format("Category: %s, Normal Kappa Mean:%.3f, Var:%.3f\n", category, getMean(normal), getVariance(normal));
			System.out.format("Category: %s, Intra Kappa Mean:%.3f, Var:%.3f\n", category, getMean(intra), getVariance(intra));
			System.out.format("Category: %s, Inter Kappa Mean:%.3f, Var:%.3f\n", category, getMean(inter), getVariance(inter));
			*/
			
			/*System.out.format("%s, Normal Kappa Mean:%.3f, Var:%.3f\n", category, getMean(normal), getVariance(normal));
			System.out.format("%s, Intra Kappa Mean:%.3f, Var:%.3f\n", category, getMean(intra), getVariance(intra));
			System.out.format("%s, Inter Kappa Mean:%.3f, Var:%.3f\n", category, getMean(inter), getVariance(inter));
			*/
			
			System.out.format("%s & %.3f/%.5f & %.3f/%.5f & %.3f/%.5f \\\\ \n", category, getMean(normal), getVariance(normal), getMean(intra), getVariance(intra), getMean(inter), getVariance(inter));
				
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

	public static void main(String[] args) {
		
		accumulatorKappa com = new accumulatorKappa();
		
		String catgories [] = {"camera","tablet","phone", "tv"};
		
		for(String category: catgories){

			String fileName = "./survey/survey/kappa/"+category+"_kappa1.csv";
			com.readcsv(fileName, category);
		}
		
      }

	

}
