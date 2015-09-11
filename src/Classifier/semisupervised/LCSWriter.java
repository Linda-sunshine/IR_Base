package Classifier.semisupervised;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import structures._Doc;
import utils.Utils;

public class LCSWriter implements Runnable{
	ArrayList<_Doc> m_collections;
	int m_start, m_end, m_core;

	public LCSWriter(int start, int end, int core, ArrayList<_Doc> collections){
		m_start = start;
		m_end = end;
		m_core = core;
		m_collections = collections;
	}
	@Override
	public void run() {
		long start = System.currentTimeMillis();
		String filename = String.format("./data/LCS/LCS_%d", m_core);
		int count = 0;
		PrintWriter LCSPrinter;
		try {
			LCSPrinter = new PrintWriter(new File(filename));
			_Doc di, dj;
			int LCS = 0;
			for(int i=m_start; i < m_end; i++){
				for(int j=0; j < m_collections.size(); j++){
					if(j > i){//To avoid duplicates.
						di = m_collections.get(m_start);
						dj = m_collections.get(j);
						LCS = Utils.LCS2Doc(di, dj);
						count++;
						LCSPrinter.write(String.format("%d,%d,%d\n", di.getID(), dj.getID(), LCS));
					}
				}
			}
			System.out.format("[%d,%d) finished, %d pairs of LCS is written to the file %s in %.4f secs.\n", m_start, m_end, count, filename, (double)(System.currentTimeMillis()-start)/1000.0);
			LCSPrinter.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}		
}
