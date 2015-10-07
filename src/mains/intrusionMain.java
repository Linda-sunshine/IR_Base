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

public class intrusionMain {
	
	private String HTSM_array[][];
	private String HTMM_array[][];
	private String LDA_array[][];
	private String ASUM_array[][];
	private int number_of_topics= 0;
	
	private String cameraTopic[]= {"battery-pos", "screen-pos","viewfinder-pos","record-pos","mode-pos","shutter-pos","memori-pos","zoom-pos","len-pos","picture-pos","price-pos","Instruction-pos","video-pos","battery-neg", "screen-neg","viewfinder-neg","record-neg","mode-neg","shutter-neg","memori-neg","zoom-neg","len-neg","picture-neg","price-neg","Instruction-neg","video-neg"};
	private String phoneTopic[]= {"screen-pos","app-pos","price-pos","battery-pos","sound-pos","camera-pos","storage-pos","call-pos","servic-pos","cpu-pos","keyboard-pos","design-pos","text-pos","screen-neg","app-neg","price-neg","battery-neg","sound-neg","camera-neg","storage-neg","call-neg","servic-neg","cpu-neg","keyboard-neg","design-neg","text-neg"};
	private String tabletTopic[]= {"screen-pos","app-pos","price-pos","battery-pos","sound-pos","camera-pos","cpu-pos","microsd-pos","internet-pos","keyboard-pos","bluetooth-pos","usb-pos","gps-pos","servic-pos","mous-pos","screen-neg","app-neg","price-neg","battery-neg","sound-neg","camera-neg","cpu-neg","microsd-neg","internet-neg","keyboard-neg","bluetooth-neg","usb-neg","gps-neg","servic-neg","mous-neg"};
	private String tvTopic[]= {"screen-pos","price-pos","sound-pos","servic-pos","picture-pos","quality-pos","app-pos","connection-pos","screen-neg","price-neg","sound-neg","servic-neg","picture-neg","quality-neg","app-neg","connection-neg"};
	private Random r;
	
	public intrusionMain(){
		r = new Random();
	}
	
	public void doRandomization(String modelName){
		
		String array[][]=null;
		if(modelName.equalsIgnoreCase("LRHTSM"))
			array=HTSM_array;
		else if(modelName.equalsIgnoreCase("LRHTMM"))
			array=HTMM_array;
		else if(modelName.equalsIgnoreCase("LDA"))
			array=LDA_array;
		else if(modelName.equalsIgnoreCase("ASUM"))
			array=ASUM_array;
		
		for(int i=0; i<this.number_of_topics; i++){
			int wordIndexInterTopic = 5;
			int wordIndexIntraTopic = 6;
			
		
			
			int randomIndexInterTopic = 0 + r.nextInt(5);
			int randomIndexIntraTopic = 0 + r.nextInt(5);
			
			String tmp = array[randomIndexInterTopic][i];
			array[randomIndexInterTopic][i] = array[wordIndexInterTopic][i];
			array[wordIndexInterTopic][i] = tmp;
			
			tmp = array[randomIndexIntraTopic][i];
			array[randomIndexIntraTopic][i] = array[wordIndexIntraTopic][i];
			array[wordIndexIntraTopic][i] = tmp;
		}
	}
	
	
	public void readcsv(String fileName, String category, String modelName)
	{
		
		BufferedReader fileReader = null;
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;
		
		
		if(modelName.equalsIgnoreCase("LRHTSM"))
			HTSM_array = new String[7][this.number_of_topics];
		else if(modelName.equalsIgnoreCase("LRHTMM"))
			HTMM_array = new String[7][this.number_of_topics];
		else if(modelName.equalsIgnoreCase("LDA"))
			LDA_array = new String[7][this.number_of_topics];
		else if(modelName.equalsIgnoreCase("ASUM"))
			ASUM_array = new String[7][this.number_of_topics];
		
		
		
		try {
	 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			
			if(!modelName.equalsIgnoreCase("ASUM")){
			
				while ((line = fileReader.readLine()) != null) {
		 
					int topicNumber = Integer.parseInt(line.substring(line.indexOf(' ')+1,line.indexOf('(')));
					//System.out.println(topicNumber);
					line = line.substring(line.indexOf(':')+1).replaceAll("\t", "");
					//System.out.println(line);
					
					String words[] = line.split(",");
					
					for(int i=0; i<words.length;i++){
						if(modelName.equalsIgnoreCase("LRHTSM"))
							HTSM_array[i][topicNumber] = words[i];
						else if(modelName.equalsIgnoreCase("LRHTMM"))
							HTMM_array[i][topicNumber] = words[i];
						else if(modelName.equalsIgnoreCase("LDA"))
							LDA_array[i][topicNumber] = words[i];
						
					}
				}
			}
			else{
				
				int first = 0;
				int rowNumber = 0;
				
				while ((line = fileReader.readLine()) != null) {
					
					if(first == 0){
						first++;
						continue;
					}
						
					
					String words[] = line.split(",");
					
					System.out.println(line);
					
					for(int i=0; i<number_of_topics;i++){
						
					  ASUM_array[rowNumber][i] = words[i];
						
					}
					
					rowNumber++;
				}
				
				System.out.println("ROw Numner"+rowNumber );
			}
					 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

	public void writeCSV(String path, String category){
		
		PrintWriter writer;
		String [] topics = null;
		
		try{
			writer = new PrintWriter(new File(path));
			if(category.equalsIgnoreCase("camera"))
				topics = cameraTopic;
			else if(category.equalsIgnoreCase("phone"))
				topics = phoneTopic;
			else if(category.equalsIgnoreCase("tablet"))
				topics = tabletTopic;
			else if(category.equalsIgnoreCase("tv"))
				topics = tvTopic;
			
			for(int i=0; i<this.number_of_topics;i++)
				writer.write(topics[i]+",");
			writer.write("\n");
			
			for(int w=0;w<7;w++){
				for(int i=0; i<this.number_of_topics;i++)
					writer.write(LDA_array[w][i]+",");
				writer.write("\n");
			}
			
			writer.write("\n\n\n\n");
			for(int i=0; i<this.number_of_topics;i++)
				writer.write(topics[i]+",");
			writer.write("\n");
			for(int w=0;w<7;w++){
				for(int i=0; i<this.number_of_topics;i++)
					writer.write(HTSM_array[w][i]+",");
				writer.write("\n");
			}
			writer.write("\n\n\n\n");
			
			for(int i=0; i<this.number_of_topics;i++)
				writer.write(topics[i]+",");
			writer.write("\n");
			for(int w=0;w<7;w++){
				for(int i=0; i<this.number_of_topics;i++)
					writer.write(HTMM_array[w][i]+",");
				writer.write("\n");
			}
			writer.write("\n\n\n\n");
			
			for(int i=0; i<this.number_of_topics;i++)
				writer.write(topics[i]+",");
			writer.write("\n");
			for(int w=0;w<7;w++){
				for(int i=0; i<this.number_of_topics;i++)
					writer.write(ASUM_array[w][i]+",");
				writer.write("\n");
			}
			writer.write("\n\n\n\n");
			
			writer.flush();
			writer.close();
			
			
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
		
	}
	
	
	public static void main(String[] args) {
		
		intrusionMain com = new intrusionMain();
		
		String modelNames[] = {"LRHTMM","LRHTSM","LDA","ASUM"}; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		String category = "tablet";
		int number_of_topics = 0;
		
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;
		
		for(String modelName:modelNames){
		
			String wordIntrusionFilePath = "./survey/"+ modelName +"_" +category+"_Topics_" + number_of_topics + "_WordIntrusion.txt";
			com.readcsv(wordIntrusionFilePath, category, modelName);
			
		}
		
		String outputFileResult = "./survey/"+category+"_Result.csv";
		com.writeCSV(outputFileResult, category);
		
		for(String modelName:modelNames){
			com.doRandomization(modelName);
		}
		
		String outputFileSurvey = "./survey/"+category+"_Survey.csv";
		com.writeCSV(outputFileSurvey, category);
		//System.out.println("Model:"+modelName+" Done");
		
      }

	

}
