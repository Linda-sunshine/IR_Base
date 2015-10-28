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

public class intrusionMeasure {
	
	private String HTSM_array[][];
	private String HTMM_array[][];
	private String LDA_array[][];
	private String ASUM_array[][];
	
	private int totalWords = 0;
	private int LDAtotalWord = 0;
	private int HTSMtotalWord = 0;
	private int HTMMtotalWord = 0;
	private int ASUMtotalWord = 0;
	
	
	private int LDA_score[][]; // MP_k_m
	private int HTSM_score[][]; // MP_k_m
	private int HTMM_score[][]; // MP_k_m
	private int ASUM_score[][]; // MP_k_m
	
	private double LDA_Intra_sum = 0;
	private double LDA_Inter_sum = 0;
	private double HTSM_Intra_sum = 0;
	private double HTSM_Inter_sum = 0;
	private double HTMM_Intra_sum = 0;
	private double HTMM_Inter_sum = 0;
	private double ASUM_Intra_sum = 0;
	private double ASUM_Inter_sum = 0;
	
	private int number_of_topics= 0;
	
	public intrusionMeasure(){
		
	}
	
	public void setNumberOfTopics(int numberTopics){
		this.number_of_topics = numberTopics;
		
		LDA_score = new int [this.number_of_topics][2];
		HTSM_score = new int [this.number_of_topics][2];
		
		HTMM_score = new int [this.number_of_topics][2];
		ASUM_score = new int [this.number_of_topics][2];
		for(int k=0;k<this.number_of_topics;k++){
			LDA_score[k][0] = 0;
			LDA_score[k][1] = 0;
			
			HTSM_score[k][0] = 0;
			HTSM_score[k][1] = 0;
			
			HTMM_score[k][0] = 0;
			HTMM_score[k][1] = 0;
			
			ASUM_score[k][0] = 0;
			ASUM_score[k][1] = 0;
		}
			
		
	}
	
	public void accumulateScore(int annotator, String category){
		for(int k =0; k<this.number_of_topics;k++){
			LDA_Intra_sum = LDA_Intra_sum+LDA_score[k][0];
			LDA_Inter_sum = LDA_Inter_sum+LDA_score[k][1];
			
			HTSM_Intra_sum = HTSM_Intra_sum+HTSM_score[k][0];
			HTSM_Inter_sum = HTSM_Inter_sum+HTSM_score[k][1];
			
			
			HTMM_Intra_sum = HTMM_Intra_sum+HTMM_score[k][0];
			HTMM_Inter_sum = HTMM_Inter_sum+HTMM_score[k][1];
			
			
			ASUM_Intra_sum = ASUM_Intra_sum+ASUM_score[k][0];
			ASUM_Inter_sum = ASUM_Inter_sum+ASUM_score[k][1];
		}
		
		//ASUM_Intra_sum = ASUM_Intra_sum>10?1:0;
		
		/*System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/normalizer,LDA_Inter_sum);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/normalizer,HTMM_Inter_sum);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/normalizer,ASUM_Inter_sum);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/normalizer,HTSM_Inter_sum);
		*/
		
		/*System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/normalizer,LDA_Inter_sum/normalizer);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/normalizer,HTMM_Inter_sum/normalizer);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/normalizer,ASUM_Inter_sum/normalizer);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/normalizer,HTSM_Inter_sum/normalizer);
		*/
		
		System.out.println("Total Word:"+ this.totalWords);
		System.out.println("LDA Total Word:"+ this.LDAtotalWord);
		
		System.out.println("HTSM Total Word:"+ this.HTSMtotalWord);
		System.out.println("HTMM Total Word:"+ this.HTMMtotalWord);
		
		System.out.println("ASUM Total Word:"+ this.ASUMtotalWord);
		
		
		//this.totalWords = this.totalWords * this.number_of_topics;
		/*
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/this.totalWords,LDA_Inter_sum/this.totalWords);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/this.totalWords,HTMM_Inter_sum/this.totalWords);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/this.totalWords,ASUM_Inter_sum/this.totalWords);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/this.totalWords,HTSM_Inter_sum/this.totalWords);
		*/
		
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum,LDA_Inter_sum);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum,HTMM_Inter_sum);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum,ASUM_Inter_sum);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum,HTSM_Inter_sum);
		
		
		System.out.format("LDA,score, %.3f\n",(LDA_Intra_sum + LDA_Inter_sum)/this.LDAtotalWord);
		System.out.format("HTMM,score, %.3f\n",(HTMM_Intra_sum+HTMM_Inter_sum)/this.HTMMtotalWord);
		System.out.format("ASUM,score, %.3f\n",(ASUM_Intra_sum+ASUM_Inter_sum)/this.ASUMtotalWord);
		System.out.format("HTSM,score, %.3f\n",(HTSM_Intra_sum+HTSM_Inter_sum)/this.HTSMtotalWord);

		
		

		System.out.println("Precision");
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/this.LDAtotalWord,LDA_Inter_sum/this.LDAtotalWord);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/this.HTMMtotalWord,HTMM_Inter_sum/this.HTMMtotalWord);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/this.ASUMtotalWord,ASUM_Inter_sum/this.ASUMtotalWord);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/this.HTSMtotalWord,HTSM_Inter_sum/this.HTSMtotalWord);
		
		System.out.println("Recall");
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/(this.number_of_topics*annotator),LDA_Inter_sum/(this.number_of_topics*annotator));
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/(this.number_of_topics*annotator),HTMM_Inter_sum/(this.number_of_topics*annotator));
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/(this.number_of_topics*annotator),ASUM_Inter_sum/(this.number_of_topics*annotator));
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/(this.number_of_topics*annotator),HTSM_Inter_sum/(this.number_of_topics*annotator));
		
		
		
		System.out.println("Intra Topic");
		System.out.println("Precision Table");
		System.out.format("%s & %.3f & %.3f & %.3f & %.3f \\\\\n", category, LDA_Intra_sum/this.LDAtotalWord, HTMM_Intra_sum/this.HTMMtotalWord, ASUM_Intra_sum/this.ASUMtotalWord, HTSM_Intra_sum/this.HTSMtotalWord);
		System.out.println("Recall Table");
		System.out.format("%s & %.3f & %.3f & %.3f & %.3f \\\\\n", category, LDA_Intra_sum/(this.number_of_topics*annotator), HTMM_Intra_sum/(this.number_of_topics*annotator), ASUM_Intra_sum/(this.number_of_topics*annotator), HTSM_Intra_sum/(this.number_of_topics*annotator));
		
		
		System.out.println("Inter Topic");
		System.out.println("Precision Table");
		System.out.format("%s & %.3f & %.3f & %.3f & %.3f \\\\\n", category, LDA_Inter_sum/this.LDAtotalWord, HTMM_Inter_sum/this.HTMMtotalWord, ASUM_Inter_sum/this.ASUMtotalWord, HTSM_Inter_sum/this.HTSMtotalWord);
		System.out.println("Recall Table");
		System.out.format("%s & %.3f & %.3f & %.3f & %.3f \\\\\n", category, LDA_Inter_sum/(this.number_of_topics*annotator), HTMM_Inter_sum/(this.number_of_topics*annotator), ASUM_Inter_sum/(this.number_of_topics*annotator), HTSM_Inter_sum/(this.number_of_topics*annotator));
		
		System.out.println(HTSM_Inter_sum+";" +this.number_of_topics*annotator+ "a"+ annotator);
		
		
		
		
		
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
					
					//System.out.println(line);
					
					for(int i=0; i<number_of_topics;i++){
						
					  ASUM_array[rowNumber][i] = words[i];
						
					}
					
					rowNumber++;
				}
				
				//System.out.println("ROw Numner"+rowNumber );
			}
					 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

	public void readSurvey(String fileName, int part, int annotatorNumber, String category){
		
		BufferedReader fileReader = null;
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;

		int startTopic = 0;
		int endTopic = number_of_topics;
		
		if(part==1){
			startTopic = 0;
			endTopic = number_of_topics/2 - 1;
 		}else{
 			startTopic = number_of_topics/2;
			endTopic = number_of_topics - 1;
 		}

		try {

			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			int lineCounter = 0;
			//int checkPoint = 10;
			while ((line = fileReader.readLine()) != null) {
				
				if(lineCounter==10){ // LDA Answer 1

					String [] words = line.split(",");
					int topicIndex = startTopic;
					//System.out.println("LDA 1,"+line);
					
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = LDA_array[5][topicIndex].trim();
						String interTopicWord = LDA_array[6][topicIndex].trim();
						//System.out.println("LDA real Intra Word:"+intraTopicWord);
						//System.out.println("LDA real Inter Word:"+interTopicWord);
						//System.out.println("Survey word:"+words[i]);
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.LDAtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						
						if(words[i]!="" || words[i]!=null){
							
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
							{
								//System.out.println("LDA Intra:"+ topicIndex+", word:"+words[i]);
								LDA_score[topicIndex][0] = LDA_score[topicIndex][0] + 1;
							}
							else if(words[i].trim().equalsIgnoreCase(interTopicWord)){
								//System.out.println("LDA Inter:"+ topicIndex+", word:"+words[i]);
								
								LDA_score[topicIndex][1] = LDA_score[topicIndex][1] + 1;
							}
						}
						
						topicIndex++;
					}
					
				}else if(lineCounter==11){ // LDA Answer 2
					
					System.out.println("LDA 2,"+line);
					String [] words = line.split(",");
					int topicIndex = startTopic;
					
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = LDA_array[5][topicIndex].trim();
						String interTopicWord = LDA_array[6][topicIndex].trim();
						

						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.LDAtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						
						if(words[i]!="" || words[i]!=null){
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								LDA_score[topicIndex][0] = LDA_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								LDA_score[topicIndex][1] = LDA_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
					
				}else if(lineCounter==22){ // HTSM Answer 1
					
					System.out.println("HTSM 1,"+line);
					String [] words = line.split(",");
					int topicIndex = startTopic;
					
					
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = HTSM_array[5][topicIndex].trim();
						String interTopicWord = HTSM_array[6][topicIndex].trim();
						//System.out.print(words[i]+",");
						

						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.HTSMtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						
						if(words[i]!="" || words[i]!=null){

							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								HTSM_score[topicIndex][0] = HTSM_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								HTSM_score[topicIndex][1] = HTSM_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
					System.out.println();
					
					
				}else if(lineCounter==23){ // HTSM Answer 2
					
					String [] words = line.split(",");
					int topicIndex = startTopic;
					System.out.println("HTSM 2,"+line);
				
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = HTSM_array[5][topicIndex].trim();
						String interTopicWord = HTSM_array[6][topicIndex].trim();
						

						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.HTSMtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						if(words[i]!="" || words[i]!=null){
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								HTSM_score[topicIndex][0] = HTSM_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								HTSM_score[topicIndex][1] = HTSM_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
				}else if(lineCounter==34){ // HTMM Answer 1
					
					String [] words = line.split(",");
					int topicIndex = startTopic;
					System.out.println("HTMM 1,"+line);
					
			
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = HTMM_array[5][topicIndex].trim();
						String interTopicWord = HTMM_array[6][topicIndex].trim();
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.HTMMtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						if(words[i]!="" || words[i]!=null){
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								HTMM_score[topicIndex][0] = HTMM_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								HTMM_score[topicIndex][1] = HTMM_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
				}else if(lineCounter==35){ // HTMM Answer 2
					
					String [] words = line.split(",");
					int topicIndex = startTopic;
					System.out.println("HTMM 2,"+line);
					
					
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = HTMM_array[5][topicIndex].trim();
						String interTopicWord = HTMM_array[6][topicIndex].trim();
						

						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.HTMMtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						if(words[i]!="" || words[i]!=null){
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								HTMM_score[topicIndex][0] = HTMM_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								HTMM_score[topicIndex][1] = HTMM_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
				}else if(lineCounter==46){ // ASUM Answer 1
					
					String [] words = line.split(",");
					int topicIndex = startTopic;
					System.out.println("ASUM 1,"+line);
					
				
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = ASUM_array[5][topicIndex].trim();
						String interTopicWord = ASUM_array[6][topicIndex].trim();
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.ASUMtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						if(words[i]!="" || words[i]!=null){
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								ASUM_score[topicIndex][0] = ASUM_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								ASUM_score[topicIndex][1] = ASUM_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
				}else if(lineCounter==47){ // ASUM Answer 2
					
					String [] words = line.split(",");
					int topicIndex = startTopic;
					System.out.println("ASUM 2,"+line);
					
				
					for(int i = 1; i<words.length;i++){
						String intraTopicWord = ASUM_array[5][topicIndex].trim();
						String interTopicWord = ASUM_array[6][topicIndex].trim();
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							this.totalWords++;
							this.ASUMtotalWord++;
							System.out.println("Words:"+words[i].trim()+", index:"+this.totalWords);
						}
						
						if(words[i]!="" || words[i]!=null){
							if(words[i].trim().equalsIgnoreCase(intraTopicWord))
								ASUM_score[topicIndex][0] = ASUM_score[topicIndex][0] + 1;
							else if(words[i].trim().equalsIgnoreCase(interTopicWord))
								ASUM_score[topicIndex][1] = ASUM_score[topicIndex][1] + 1;
						}
						topicIndex++;
					}
				}
				lineCounter++;
			}

			// Show the scores
			

		}catch(Exception e){
			System.err.println("File "+ fileName+" Not Found!!");
			e.printStackTrace();
		}
	}
	
		
	public static void main(String[] args) {
		
		intrusionMeasure com = new intrusionMeasure();
		
		String modelNames[] = {"LRHTMM","LRHTSM","LDA","ASUM"}; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		String category = "tablet";
		int number_of_topics = 0;
		int partLimit = 2;
		
		
		
		if(category.equalsIgnoreCase("tablet"))
		{
			number_of_topics = 30;
		}
		else if(category.equalsIgnoreCase("camera"))
		{
			number_of_topics = 26;
		}
		else if(category.equalsIgnoreCase("phone"))
		{
			number_of_topics = 26;
		}
		else if(category.equalsIgnoreCase("tv"))
		{
			number_of_topics = 16;
			partLimit = 1;
		}
		
		for(String modelName:modelNames){
			String wordIntrusionFilePath = "./survey/"+ modelName +"_" +category+"_Topics_" + number_of_topics + "_WordIntrusion.txt";
			com.readcsv(wordIntrusionFilePath, category, modelName);
			
		}
		
		String filePath = "./survey/survey/result/";
		com.setNumberOfTopics(number_of_topics);
		for(int part = 1;part <=partLimit; part++){
		
			for(int annotatorNumber = 1;annotatorNumber<=3;annotatorNumber++)
			{
				String fileName = filePath+category+"_"+part+"_"+annotatorNumber+".csv";
				com.readSurvey(fileName, part, annotatorNumber, category);
			}
		
		}
		
		//int totalCount = (number_of_topics)*3; // # of topics by 2 * #of annotator
		com.accumulateScore(3, category);
				
      }

	

}
