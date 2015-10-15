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
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.Set;

import org.netlib.util.booleanW;
import org.netlib.util.doubleW;

public class summaryMeasureNewKappa {
	
	private int number_of_topics= 0;
	private String category;
	private int normalmatrix [][] = {{0,0},{0,0}};
	private double kappa [] ;
	
	
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
	
	public void setCategory (String category){
		this.category = category;
		if(category.equalsIgnoreCase("tv")){
			kappa = new double[3];
		}else{
			kappa = new double[15];
		}
		
	}
	
	public void readSummarySurvey(String fileName, String pairFile, int index){
		BufferedReader fileReader = null;
		BufferedReader pairReader = null;
		
		try {
			 
			fileReader = new BufferedReader(new FileReader(fileName));
			pairReader = new BufferedReader(new FileReader(pairFile));
			String line, pairLine;
			int lineCounter = 0;
			int c = 0;
			int booleanMatrix [][] = new int [6][2];
			for(int a=0; a<6;a++){
				for(int b=0;b<2;b++){
					booleanMatrix[a][b] = 0;
				}
			}
			ArrayList<String> list = new ArrayList<String>();
			
			while ((line = fileReader.readLine()) != null &&  (pairLine = pairReader.readLine()) != null) {

				
				
				if(lineCounter>=3+c && lineCounter<=8+c){
					String sentence = line.split(",")[1].trim();
					//System.out.println(sentence);
					list.add(sentence);
					if(lineCounter==8+c)
						c = c + 13;
				}
				
				
				
				
				if(line.contains("Answer 1")){
					//System.out.println(" Owner Answer 1");
					if(line.split(",").length>1){
						String summary = line.split(",")[1].trim();
						//System.out.println(summary);
							for(int i=0; i<6; i++){
								if(list.get(i).equalsIgnoreCase(summary)){
									//System.out.println("Found:"+summary);
									booleanMatrix[i][0] = 1;
									break;
								}
							}
					}
						
					}
				
				if(pairLine.contains("Answer 1")){
					//System.out.println(" Pair Answer 1");
					if(pairLine.split(",").length>1){
						String summary = pairLine.split(",")[1].trim();
						
							for(int i=0; i<6; i++){
								if(list.get(i).equalsIgnoreCase(summary)){
									//System.out.println("Found:"+summary);
									booleanMatrix[i][1] = 1;
									break;
								}
						}
					
				}
				}
				
				
				if(line.contains("Answer 2")){
					//System.out.println(" Owner Answer 2");
					if(line.split(",").length>1){
						String summary = line.split(",")[1].trim();
						
							for(int i=0; i<6; i++){
								if(list.get(i).equalsIgnoreCase(summary)){
									//System.out.println("Found:"+summary);
									booleanMatrix[i][0] = 1;
									break;
								}
							}
					}
				}
				if(pairLine.contains("Answer 2")){
					//System.out.println(" Pair Answer 2");
					if(pairLine.split(",").length>1){
						String summary = pairLine.split(",")[1].trim();
						
							for(int i=0; i<6; i++){
								if(list.get(i).equalsIgnoreCase(summary)){
									//System.out.println("Found:"+summary);
									booleanMatrix[i][1] = 1;
									break;
								}
							}
					}
					
					
					//accumulate count
					for(int j=0; j<6;j++){
						normalmatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
					}
					// now clear all
					for(int a=0; a<6;a++){
						for(int b=0;b<2;b++){
							booleanMatrix[a][b] = 0;
						}
					}
					// clear the list
					list.clear();
					
				}
				lineCounter++;
			}
			
			/*for(int a=0;a<2;a++){
				for(int b=0;b<2;b++){
					System.out.print(normalmatrix[a][b]+",");
				}
			}
			
			System.out.println();
			*/
			int total = normalmatrix[0][0] + normalmatrix[0][1] + normalmatrix[1][0] +  normalmatrix[1][1] ;
			double p_o = (double)(normalmatrix[0][0] + normalmatrix[1][1])/total;
			int firstRowSum = normalmatrix[0][0] + normalmatrix[0][1];
			int secondRowSum = normalmatrix[1][0] + normalmatrix[1][1];
			
			int firstColSum = normalmatrix[0][0] + normalmatrix[1][0];
			int secondColSum = normalmatrix[0][1] + normalmatrix[1][1];
			
			double A_no_p = (double) firstRowSum / total;
			double A_yes_p = (double) secondRowSum / total;
			
			double B_no_p = (double) firstColSum / total;
			double B_yes_p = (double) secondColSum / total;
			double p_e = (A_yes_p*B_yes_p) + (A_no_p*B_no_p);
			
			double normal_kappa = (p_o - p_e)/(1 - p_e);
			//System.out.print("Normal Kappa,"+normal_kappa+"\n");
			kappa[index] = normal_kappa;
			
			
			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
		
		
	}
	
	public void printKappa(){
		System.out.format("%s & %.3f/%.5f \\\\ \n", category, getMean(kappa), getVariance(kappa));
	}
	
	public static void main(String[] args) {
		
		
		
		String catgories [] = {"camera","tablet","phone", "tv"};
		for(String category: catgories){
			summaryMeasureNewKappa com = new summaryMeasureNewKappa();
			com.setCategory(category);
			int t = 6;

			if(category.equalsIgnoreCase("tv"))
				t = 3;

			int counter = 0;

			for(int i=1; i<t; i++){
				String fileName = "./survey/survey/summaryResult/"+category+"_"+i+".csv";
				for(int j=i+1; j<=t; j++){
					//System.out.println("pair ("+i+","+j+")");
					String pairfile = "./survey/survey/summaryResult/"+category+"_"+j+".csv";
					com.readSummarySurvey(fileName, pairfile, counter);
					counter++;
				}

			}

			com.printKappa();
			//System.out.println("Counter:"+ counter);
		}
		}
	
	

	

}
