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

public class summaryMain {
	
	
	private HashMap<String,String> HTSM_summary = new HashMap<String, String>();
	private HashMap<String,String> HTMM_summary = new HashMap<String, String>();
	private HashMap<String,String> ASUM_summary = new HashMap<String, String>();
	
	private HashMap<String, ArrayList<String>> list =new HashMap<String, ArrayList<String>>();
	
	
	private int number_of_topics= 0;
	private int number_of_Total_sentences = 6;
	
	/*
	private String cameraTopic[]= {"battery-pos", "screen-pos","viewfinder-pos","record-pos","mode-pos","shutter-pos","memori-pos","zoom-pos","len-pos","picture-pos","price-pos","Instruction-pos","video-pos","battery-neg", "screen-neg","viewfinder-neg","record-neg","mode-neg","shutter-neg","memori-neg","zoom-neg","len-neg","picture-neg","price-neg","Instruction-neg","video-neg"};
	private String phoneTopic[]= {"screen-pos","app-pos","price-pos","battery-pos","sound-pos","camera-pos","storage-pos","call-pos","servic-pos","cpu-pos","keyboard-pos","design-pos","text-pos","screen-neg","app-neg","price-neg","battery-neg","sound-neg","camera-neg","storage-neg","call-neg","servic-neg","cpu-neg","keyboard-neg","design-neg","text-neg"};
	private String tabletTopic[]= {"screen-pos","app-pos","price-pos","battery-pos","sound-pos","camera-pos","cpu-pos","microsd-pos","internet-pos","keyboard-pos","bluetooth-pos","usb-pos","gps-pos","servic-pos","mous-pos","screen-neg","app-neg","price-neg","battery-neg","sound-neg","camera-neg","cpu-neg","microsd-neg","internet-neg","keyboard-neg","bluetooth-neg","usb-neg","gps-neg","servic-neg","mous-neg"};
	private String tvTopic[]= {"screen-pos","price-pos","sound-pos","servic-pos","picture-pos","quality-pos","app-pos","connection-pos","screen-neg","price-neg","sound-neg","servic-neg","picture-neg","quality-neg","app-neg","connection-neg"};
	*/

	private String cameraTopic[]= {"battery", "screen","viewfinder","record","mode","shutter","memori","zoom","len","picture","price","Instruction","video","battery", "screen","viewfinder","record","mode","shutter","memori","zoom","len","picture","price","Instruction","video"};
	private String phoneTopic[]= {"screen","app","price","battery","sound","camera","storage","call","servic","cpu","keyboard","design","text","screen","app","price","battery","sound","camera","storage","call","servic","cpu","keyboard","design","text"};
	private String tabletTopic[]= {"screen","app","price","battery","sound","camera","cpu","microsd","internet","keyboard","bluetooth","usb","gps","servic","mous","screen","app","price","battery","sound","camera","cpu","microsd","internet","keyboard","bluetooth","usb","gps","servic","mous"};
	private String tvTopic[]= {"screen","price","sound","servic","picture","quality","app","connection","screen","price","sound","servic","picture","quality","app","connection"};

	
	private Random r;
	
	String tabletProductList[] = {"B008DWG5HE","B00CYQPM42","B007P4YAPK"};
	String cameraProductList[] = {"B005IHAIMA","B002IPHIEG","B00DMS0LCO"};
	String phoneProductList[] = {"B00COYOAYW","B004T36GCU","B008HTJLF6"};
	String tvProductList[] = {"B0074FGLUM","B00BCGROJG","B00AOA9BL0"};
	
	HashMap<String,String> tabletProductNames = new HashMap<String,String>();
	HashMap<String,String> phoneProductNames = new HashMap<String,String>();
	HashMap<String,String> cameraProductNames = new HashMap<String,String>();
	HashMap<String,String> tvProductNames = new HashMap<String,String>();
	
	
	//	{"Samsung Galaxy Note 10.1","Amazon Kindle Fire HDX","ASUS Transformer Tablet"};
	
	
	
	private PrintWriter resultWriter;
	private PrintWriter surveyWriter;
	
	public summaryMain(){
		r = new Random();
		
		tabletProductNames.put("B008DWG5HE", "Samsung Galaxy Note 10.1");
		tabletProductNames.put("B00CYQPM42", "Amazon Kindle Fire HDX");
		tabletProductNames.put("B007P4YAPK", "ASUS Transformer Tablet");
		
		phoneProductNames.put("B00COYOAYW", "Nokia Lumia 521");
		phoneProductNames.put("B004T36GCU", "HTC A9192 Phone");
		phoneProductNames.put("B008HTJLF6", "Samsung Galaxy S III");
		
		cameraProductNames.put("B005IHAIMA", "Sony NEX-5N 16.1 MP Compact Interchangeable Lens Camera");
		cameraProductNames.put("B002IPHIEG", "Sony Cybershot DSC");
		cameraProductNames.put("B00DMS0LCO", "Canon EOS 70D Digital SLR Camera");
		
		tvProductNames.put("B0074FGLUM", "Samsung UN50EH5300 50-Inch 1080p 60Hz LED HDTV");
		tvProductNames.put("B00BCGROJG", "Samsung Ultra Slim Smart LED HDTV 2013 Model");
		tvProductNames.put("B00AOA9BL0", "XFINITY TV Go");
	}
	
	public void setResultWriter(String path){
		try{
			resultWriter = new PrintWriter(new File(path));
			
		}
		catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	
public void writeCSV(String path, String category, int[] topicID){
		
		PrintWriter writer = null;
		String [] topics = null;
		HashMap<String,String> productNames = new HashMap<String,String>();
		try{
			writer = new PrintWriter(new File(path));
		}
		catch(Exception e){
			System.err.println(path+" Not found!!");
			e.printStackTrace();
			
		}
			if(category.equalsIgnoreCase("camera"))
				topics = cameraTopic;
			else if(category.equalsIgnoreCase("phone"))
				topics = phoneTopic;
			else if(category.equalsIgnoreCase("tablet"))
				topics = tabletTopic;
			else if(category.equalsIgnoreCase("tv"))
				topics = tvTopic;
			
			
			String productList [] = null;
			if(category.equalsIgnoreCase("camera")){
				topics = cameraTopic;
				productList = cameraProductList;
				productNames = cameraProductNames;
			}
			else if(category.equalsIgnoreCase("phone")){
				topics = phoneTopic;
				productList = phoneProductList;
				productNames = phoneProductNames;
			}
			else if(category.equalsIgnoreCase("tablet")){
				topics = tabletTopic;
				productList = tabletProductList;
				productNames = tabletProductNames;
			}
			else if(category.equalsIgnoreCase("tv")){
				topics = tvTopic;
				productList = tvProductList;
				productNames = tvProductNames;
			}
			
			//writer.write("Under each cate");
			for(String prodID:productList){
				
				
				for(int topic = 0; topic<topicID.length; topic++){
					int posTopic = topicID[topic];
					writer.write("Product name:"+productNames.get(prodID)+"\n");
					writer.write("Topic ID:"+ posTopic+"\n");
					writer.write("Tell me something POSITIVE about "+ topics[posTopic].toUpperCase()+" Aspect \n");
					String key = prodID+posTopic;
					ArrayList<String> sentences = list.get(key);
					int sentenceCounter = 0;
					if(sentences==null)
						System.out.println("NULL");
					for(String sentence:sentences){
						System.out.println("sen:"+sentence);
						writer.write(sentenceCounter+","+sentence+"\n");
						sentenceCounter++;
					}
					
					writer.write("Answer 1:,\n");
					writer.write("Answer 2:,\n");
					
					writer.write("\n\n");
					
					writer.write("Product name:"+productNames.get(prodID)+"\n");
					int negTopic = topicID[topic] + this.number_of_topics/2;
					writer.write("Topic ID:"+ negTopic+"\n");
					writer.write("Tell me something NEGATIVE about "+ topics[posTopic].toUpperCase()+" Aspect \n");
					
					key = prodID+negTopic;
					sentences = list.get(key);
					sentenceCounter = 0;
					for(String sentence:sentences){
						writer.write(sentenceCounter+","+sentence+"\n");
						sentenceCounter++;
					}
					writer.write("Answer 1:,\n");
					writer.write("Answer 2:,\n");
					
					writer.write("\n\n");
					
				}
				
			}
			
			writer.flush();
			writer.close();
			
			
		
}
	
	
	public void doInterleaving(String category, int [] topicID){
		
		String topics [] = null;
		String productList [] = null;
		if(category.equalsIgnoreCase("camera")){
			topics = cameraTopic;
			productList = cameraProductList;
		}
		else if(category.equalsIgnoreCase("phone")){
			topics = phoneTopic;
			productList = phoneProductList;
		}
		else if(category.equalsIgnoreCase("tablet")){
			topics = tabletTopic;
			productList = tabletProductList;
		}
		else if(category.equalsIgnoreCase("tv")){
			topics = tvTopic;
			productList = tvProductList;
		}
		
		for(String prodID:productList){
			System.out.println(prodID);
			for(int topic = 0; topic<topicID.length; topic++){
			int posTopic = topicID[topic];
			int negTopic = topicID[topic] + this.number_of_topics/2;
			
			int HTSM_counter = 0;
			int HTSM_index = 0;
			int HTMM_counter = 0;
			int HTMM_index = 0;
			int ASUM_counter = 0;
			int ASUM_index = 0;
			
			while(HTSM_counter<2 || HTMM_counter<2 || ASUM_counter<2){
				
				int randomNUmber = 0 + r.nextInt(3);
				
				if(HTSM_counter<2 && randomNUmber==0){
					boolean flag = false;
					while(HTSM_index<25 && flag==false){
						System.out.print("Pos Topic:"+ posTopic+", ");
						System.out.println("HTSM Index:"+HTSM_index);
						String key = prodID+posTopic+HTSM_index;
						String sentence = HTSM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTSM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
							flag = true;
							HTSM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
								
								flag = true;
								HTSM_counter++;
								break;
							}
						}
						HTSM_index++;
					}
					
				}
				
				if(HTMM_counter<2 && randomNUmber==1){
					boolean flag = false;
					while(HTMM_index<25 && flag==false){
						System.out.print("Pos Topic:"+ posTopic+", ");
						System.out.println("HTMM Index:"+HTMM_index);
						String key = prodID+posTopic+HTMM_index;
						String sentence = HTMM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTMM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
							
							flag = true;
							HTMM_counter++;
							break;
						}
						else {
							//System.out.println("IN HTMM contains key :");
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
								
								flag = true;
								HTMM_counter++;
								break;
							}
						}
						HTMM_index++;
					}
					
				}
				
				if(ASUM_counter<2 && randomNUmber==2){
					boolean flag = false;
					while(ASUM_index<25 && flag==false){
						System.out.print("Pos Topic:"+ posTopic+", ");
						System.out.println("ASUM Index:"+ASUM_index);
						String key = prodID+posTopic+ASUM_index;
						String sentence = ASUM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("ASUM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
							
							flag = true;
							ASUM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
								
								flag = true;
								ASUM_counter++;
								break;
							}
						}
						ASUM_index++;
					}
					
					
				}
				
			}
			
		
			//Negative Topics //
			
			posTopic = negTopic = topicID[topic] + this.number_of_topics/2;
			
			HTSM_counter = 0;
			HTSM_index = 0;
			HTMM_counter = 0;
			HTMM_index = 0;
			ASUM_counter = 0;
			ASUM_index = 0;
			
			while(HTSM_counter<2 || HTMM_counter<2 || ASUM_counter<2){
				int randomNUmber = 0 + r.nextInt(3);
				if(HTSM_counter<2 && randomNUmber==0){
					boolean flag = false;
					while(HTSM_index<25 && flag==false){
						System.out.print("Neg Topic:"+ posTopic+", ");
						System.out.println("HTSM Index:"+HTSM_index);
						String key = prodID+posTopic+HTSM_index;
						String sentence = HTSM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTSM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
							
							flag = true;
							HTSM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
								
								flag = true;
								HTSM_counter++;
								break;
							}
						}
						HTSM_index++;
					}
					
				}
				
				if(HTMM_counter<2 && randomNUmber==1){
					boolean flag = false;
					while(HTMM_index<25 && flag==false){
						System.out.print("Neg Topic:"+ posTopic+", ");
						System.out.println("HTMM Index:"+HTMM_index);
						String key = prodID+posTopic+HTMM_index;
						String sentence = HTMM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTMM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
							
							flag = true;
							HTMM_counter++;
							break;
						}
						else {
							//System.out.println("IN HTMM contains key :");
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
								
								flag = true;
								HTMM_counter++;
								break;
							}
						}
						HTMM_index++;
					}
					
				}
				
				if(ASUM_counter<2 && randomNUmber==2){
					boolean flag = false;
					while(ASUM_index<25 && flag==false){
						System.out.print("Neg Topic:"+ posTopic+", ");
						System.out.println("ASUM Index:"+ASUM_index);
						String key = prodID+posTopic+ASUM_index;
						String sentence = ASUM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("ASUM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
							
							flag = true;
							ASUM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
								
								flag = true;
								ASUM_counter++;
								break;
							}
						}
						ASUM_index++;
					}
					
					
				}
				
			}
			
			
			
			
			}
			
		}
		resultWriter.flush();
		resultWriter.close();
		
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
		
		try {
	 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			if(modelName.equalsIgnoreCase("LRHTSM")){	
			
				while ((line = fileReader.readLine()) != null) {
					String infos[] = line.split(",");
					String prodID = infos[0];
					int topicID = Integer.parseInt(infos[1]);
					int sentenceID = Integer.parseInt(infos[2]);
					String sentence = infos[3];
					
					System.out.println();
					
					String hashKey = prodID+topicID+sentenceID;
					HTSM_summary.put(hashKey, sentence);
					
					System.out.println("HTSM:"+prodID+","+topicID+","+sentenceID);
					System.out.println(HTSM_summary.get(hashKey));
					
					//line = "";
				}
			
			}else if(modelName.equalsIgnoreCase("LRHTMM")){	
			
				while ((line = fileReader.readLine()) != null) {
					String infos[] = line.split(",");
					String prodID = infos[0];
					int topicID = Integer.parseInt(infos[1]);
					int sentenceID = Integer.parseInt(infos[2]);
					String sentence = infos[3];
					
					String hashKey = prodID+topicID+sentenceID;
					HTMM_summary.put(hashKey, sentence);
					System.out.println("HTMM:"+prodID+","+topicID+","+sentenceID);
					
					System.out.println(HTMM_summary.get(hashKey));
					
					//line = "";
				}
			
			}else if(modelName.equalsIgnoreCase("ASUM")){	
			
				
				while ((line = fileReader.readLine()) != null) {
					String infos[] = line.split(",");
					String prodID = infos[0];
					int topicID = Integer.parseInt(infos[1]);
					int sentenceID = Integer.parseInt(infos[2]);
					String sentence = infos[3];
					
					String hashKey = prodID+topicID+sentenceID;
					ASUM_summary.put(hashKey, sentence);
					System.out.println("ASUM:"+prodID+","+topicID+","+sentenceID);
					
					System.out.println(ASUM_summary.get(hashKey));
					
					//line = "";
				}
			
			}

					 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

	/*public void writeCSV(String path, String category){
		
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
		
	}*/
	
	
	public static void main(String[] args) {
		
		summaryMain com = new summaryMain();
		
		String modelNames[] = {"LRHTMM","LRHTSM","ASUM"}; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		String category = "tv";
		int number_of_topics = 0;
		int tablettopicID [] = {1,2,3,7};
		int phonetopicID [] = {2,3,5,8};
		//int cameratopicID [] = {0,1,11,12};
		
		int cameratopicID [] = {1,9,10,12};
		
		int tvtopicID [] = {0,1,2,3};
		
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;
		
		for(String modelName:modelNames){
		
			String summaryFilePath = "./survey/"+ modelName +"_" +category+"_Topics_" + number_of_topics + "_Summary.txt";
			com.readcsv(summaryFilePath, category, modelName);
			
		}
		
		String outputFileResult = "./survey/"+category+"_DocSummary_Result.csv";
		com.setResultWriter(outputFileResult);
		
	    System.out.println("Reading Done");
		
		
		
		int topicID[] = null;
		
		if(category.equalsIgnoreCase("tablet"))
			topicID = tablettopicID;
		else if(category.equalsIgnoreCase("camera"))
			topicID = cameratopicID;
		else if(category.equalsIgnoreCase("phone"))
			topicID = phonetopicID;
		else if(category.equalsIgnoreCase("tv"))
			topicID = tvtopicID;
		
		
		com.doInterleaving(category, topicID);
		
		String outputFileSurvey = "./survey/"+category+"_DocSummary_Survey.csv";
		com.writeCSV(outputFileSurvey, category, topicID);
		
	    System.out.println("Done");
		
		/*
		for(String modelName:modelNames){
			com.doRandomization(modelName);
		}
		*/
		//String outputFileSurvey = "./survey/"+category+"_Survey.csv";
		//com.writeCSV(outputFileSurvey, category);
		//System.out.println("Model:"+modelName+" Done");
		
      }

	

}
