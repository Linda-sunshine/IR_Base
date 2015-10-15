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
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.Set;

public class JSTPR_sentence1 {
	
	Map<String, Double> dictionary = new HashMap<String, Double>();
	Map<String, Integer> sortedMap;
	
	HashMap<Integer, ArrayList<Double>> prosList = new HashMap<Integer, ArrayList<Double>>();
	HashMap<Integer, ArrayList<Double>> consList = new HashMap<Integer, ArrayList<Double>>();
	
	// Real 0 = pros and real 1 = cons
	// JST Predicted pred 0 = neutral, 1 = pros, 2 = cons
	// so need careful mapping
	public void readcsv(String real, String pred, int trainSize)
	{
		
		int precision_recall [][] = new int[2][2]; // row for true label and col for predicted label 
		precision_recall[0][0] = 0; // 0 is for pros
		precision_recall[0][1] = 0; // 1 is for cons 
		precision_recall[1][0] = 0;
		precision_recall[1][1] = 0;
		
		BufferedReader br_real = null;
		BufferedReader br_pred = null;
		String real_line = "";
		String pred_line = "";
	
		//System.out.println("Here");
		int c = 0;
		int lineCount = 0;
		try {
	 
			br_real = new BufferedReader(new FileReader(real));
			br_pred = new BufferedReader(new FileReader(pred));
			
			while ((real_line = br_real.readLine()) != null && (pred_line = br_pred.readLine()) != null) {
	 
				String label[] = real_line.split(" ");
				String predict[] = pred_line.split(" ");
				
				int real_val = Integer.parseInt(label[1]);
				int pred_val = Integer.parseInt(predict[1]);
				
				if(real_val == 3) break; // 3 means from Amazon so skip
				
				lineCount++;
				
				if( pred_val == 0){
					c++;
					continue; // skipping the neutral prediction of JST
				}
				
				// Real 0 = pros and real 1 = cons
				// Predicted pred 0 = neutral, 1 = pros, 2 = cons
				
				if(pred_val==1) // means it is predicted as pros
				{
					pred_val = 0; // real pros is 0 so mapping predicted 1 to real 0
				}else if (pred_val == 2){ // means it is predicted as cons
					pred_val = 1;   // real cons is 1 so mapping predicted 2 to real 1
				}
				
				// NO need to change pos as both in both fomrat 1 is pos
				
				if(real_val>1 || pred_val>1)
					System.out.println(real_val+","+pred_val);
				else
				precision_recall[real_val][pred_val]++;
			}
			/*System.out.println("Confusion Matrix");
			for(int i=0; i<2; i++)
			{
				for(int j=0; j<2; j++)
				{
					System.out.print(precision_recall[i][j]+",");
				}
				System.out.println();
			}*/
			
			
			//System.out.println("Neutral Sentence:"+ c);
			//System.out.println("NewEgg Sentence:"+ lineCount);
			
			double pros_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
			double cons_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);
			
			
			double pros_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
			double cons_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);
			
			//System.out.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
			//System.out.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);
			
			
			double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
			double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
			
			System.out.format("F1 measure:pros,cons: %.3f %.3f\n",pros_f1,cons_f1);
			if(!prosList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = prosList.get(trainSize);
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			
			
			if(!consList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = consList.get(trainSize);
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			
			pros_f1=cons_f1=0;
			
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

	
	public void readcsvHTSM(String real,int trainSize)
	{
		
		
		BufferedReader br_real = null;
		BufferedReader br_pred = null;
		String line = "";
		
		double pros_f1= 0; double cons_f1 = 0;
		try {
	 
			br_real = new BufferedReader(new FileReader(real));
			
			while ((line = br_real.readLine()) != null) {
				
				if(line.contains("F1 measure")){

					//System.out.println("Find");
					int index = line.indexOf(":");
					//System.out.println("First : index"+ index);
					int another_index = line.indexOf(":", index+1);
					//System.out.println("Second : index"+ another_index);
					int commaIndex = line.indexOf(",");
					
					int thirdSemiColonIndex = line.indexOf(":", commaIndex+1);
					pros_f1 = Double.parseDouble(line.substring(another_index+1, commaIndex));
					cons_f1 = Double.parseDouble(line.substring(thirdSemiColonIndex+1));
					
					
				}
				
			}
			
			
			System.out.format("F1 measure:pros,cons: %.3f %.3f\n",pros_f1,cons_f1);
			
			if(!prosList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = prosList.get(trainSize);
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			
			
			if(!consList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = consList.get(trainSize);
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			
			
			pros_f1= 0; 
			cons_f1 = 0;
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}
	

	public void readcsvEMNaive(String real,int trainSize)
	{
		
		
		BufferedReader br_real = null;
		BufferedReader br_pred = null;
		String line = "";
		
		double pros_f1= 0; double cons_f1 = 0;
		try {
	 
			br_real = new BufferedReader(new FileReader(real));
			
			while ((line = br_real.readLine()) != null) {
				
				if(line.contains("completeF1")){

					String [] infos = line.split(",");
					pros_f1 = Double.parseDouble(infos[1]);
					cons_f1 = Double.parseDouble(infos[2]);
					
					
				}
				
			}
			
			System.out.format("F1 measure:pros,cons: %.3f %.3f\n",pros_f1,cons_f1);
			
			if(!prosList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = prosList.get(trainSize);
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			
			
			if(!consList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = consList.get(trainSize);
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			
			
			pros_f1= 0; 
			cons_f1 = 0;
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}
	
	
	
	
	public void readcsvASUM(String real, int trainSize)
	{
		
		
		BufferedReader br_real = null;
		BufferedReader br_pred = null;
		String line = "";
		
		double pros_f1= 0; double cons_f1 = 0;
		try {
	 
			br_real = new BufferedReader(new FileReader(real));
			
			while ((line = br_real.readLine()) != null) {
				
				if(line.contains("Sentence level F1 measure")){

					int index = line.indexOf(":");
					//System.out.println("First : index"+ index);
					int another_index = line.indexOf(":", index+1);
					//System.out.println("Second : index"+ another_index);
					int commaIndex = line.indexOf(",");
					
					int thirdSemiColonIndex = line.indexOf(":", commaIndex+1);
					pros_f1 = Double.parseDouble(line.substring(another_index+1, commaIndex));
					cons_f1 = Double.parseDouble(line.substring(thirdSemiColonIndex+1));
					
					
				}
				
			}
			
			
			System.out.format("F1 measure:pros,cons: %.3f %.3f\n",pros_f1,cons_f1);
			if(!prosList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = prosList.get(trainSize);
				list.add(pros_f1);
				prosList.put(trainSize, list);
			}
			
			
			if(!consList.containsKey(trainSize)){
				ArrayList<Double> list = new ArrayList<Double>();
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			else{
				ArrayList<Double> list = consList.get(trainSize);
				list.add(cons_f1);
				consList.put(trainSize, list);
			}
			
			pros_f1=cons_f1=0;
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}
	
	
	public void writeGNUPlotfile(String category){
		String path = "/home/nahid/workspace/plot/"+category+"_pros.csv";
			try{
				PrintWriter writer = new PrintWriter(new File(path));
				writer.println("#Model\tASUM\tHTSM\tJST\tEM-NaiveBayes");
				for(int trainSize = 0; trainSize<=5000;trainSize=trainSize+1000){
					ArrayList<Double> list = prosList.get(trainSize);
					writer.write(trainSize+"\t");
					for(double d:list){
						writer.write(d+"\t");
					}
					writer.write("\n");
				}
				
				writer.flush();
				writer.close();
				
				path = "/home/nahid/workspace/plot/"+category+"_cons.csv";
				writer = new PrintWriter(new File(path));
				writer.println("#Model\tASUM\tHTSM\tJST\tEM-NaiveBayes");
				for(int trainSize = 0; trainSize<=5000;trainSize=trainSize+1000){
					ArrayList<Double> list = consList.get(trainSize);
					writer.write(trainSize+"\t");
					for(double d:list){
						writer.write(d+"\t");
					}
					writer.write("\n");
				}
				

				writer.flush();
				writer.close();
				
			}
			catch(Exception e){
				System.err.println("File:"+path+" Not Found!!!");
			}
		
		prosList.clear();
		consList.clear();
		
		
	}
	
	
	
	public static void main(String[] args) {
		
		JSTPR_sentence1 com = new JSTPR_sentence1();
		String modellist[] = {"ASUM","HTSM","JST", "EM-NB"};
		
		String categories[] = {"camera","phone","tablet","tv"};
		int trainSize = 0;
		
		
		for(String category:categories){
			
			for(String model:modellist){

				System.out.println("Model: "+model);	
				int number_of_topics = 0;

				if(model.equalsIgnoreCase("JST")){

					for(trainSize = 0; trainSize<=5000;trainSize=trainSize+1000){
						System.out.print("Category:"+category+", TrainSize: "+ trainSize+" ");
						String filename1 = "/home/nahid/workspace/JST-master/src/data/"+trainSize+"/"+ category + "/"+ "MR_test_label_sentence.dat";
						String filename2 = "/home/nahid/workspace/JST-master/src/result/"+trainSize+"/"+ category + "/senti/"+"/predicted_label_sentence.dat";
						com.readcsv(filename1,filename2,trainSize);
					}

				}

				if(model.equalsIgnoreCase("ASUM")){

					for(trainSize = 0; trainSize<=5000;trainSize=trainSize+1000){
						System.out.print("Category:"+category+", TrainSize: "+ trainSize+" ");
						String filename1 = "/media/nahid/01CFEA0AB76A6710/git/ASUM/ASUM/data/output/"+trainSize+"/"+ category + "/"+category+"_information.txt";
						com.readcsvASUM(filename1,trainSize);
					}

				}

				if(model.equalsIgnoreCase("HTSM")){


					if(category.equalsIgnoreCase("tablet"))
						number_of_topics = 30;
					else if(category.equalsIgnoreCase("camera"))
						number_of_topics = 26;
					else if(category.equalsIgnoreCase("phone"))
						number_of_topics = 26;
					else if(category.equalsIgnoreCase("tv"))
						number_of_topics = 16;

					for(trainSize = 0; trainSize<=5000;trainSize=trainSize+1000){
						System.out.print("Category:"+category+", TrainSize: "+ trainSize+" ");
						String filename1 = "/home/nahid/workspace/Nahid_IR_Base/model/LRHTSM_senti/result/"+category+"/"+trainSize+"/NewEggLoaded/aspectSentiPrior/"+"Topics_"+number_of_topics+"_Information.txt";
						com.readcsvHTSM(filename1,trainSize);
					}
				}
				
				if(model.equalsIgnoreCase("EM-NB")){
					if(category.equalsIgnoreCase("tablet"))
						number_of_topics = 30;
					else if(category.equalsIgnoreCase("camera"))
						number_of_topics = 26;
					else if(category.equalsIgnoreCase("phone"))
						number_of_topics = 26;
					else if(category.equalsIgnoreCase("tv"))
						number_of_topics = 16;
					
					
					for(trainSize = 0; trainSize<=5000;trainSize=trainSize+1000){
						System.out.print("Category:"+category+", TrainSize: "+ trainSize+" ");
						String resultPath = "./model/"+model+"/result/"+category+"/"+trainSize+"/"+category+"_information.txt";
						com.readcsvEMNaive(resultPath,trainSize);
					}
					
				}
			}
			
			com.writeGNUPlotfile(category);
		
		}
		
      }

	

}
