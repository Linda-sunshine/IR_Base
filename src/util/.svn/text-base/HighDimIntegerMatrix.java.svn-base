package util;

public class HighDimIntegerMatrix {
	public int dim;
	public int length;
	Node<Integer> memory;
	
	public HighDimIntegerMatrix(int dim, int length) {
		this.dim = dim;
		this.memory = buildTree(dim, length);
	}
	
	public Node<Integer> buildTree(int dim, int numBranches) {
		return buildTree(dim, numBranches, dim);
	}
	
	public Node<Integer> buildTree(int dim, int numBranches, int currentOrder) {
		Node<Integer> root = new Node<Integer>();
		if (currentOrder != 0) {
			for (int i = 0; i < numBranches; i++) {
				root.addLink(buildTree(dim, numBranches, currentOrder-1));
			}
		}
		return root;
	}
	
	public Integer get(Iterable<Integer> index) throws Exception {
		return memory.get(index).getValue();
	}
	
	public Integer get(Iterable<Integer> index, int key) throws Exception {
		return memory.get(index).getValue(key);
	}
	
	public void increase(Iterable<Integer> index, int value) {
		memory.get(index).value += value;
	}
	
	public void decrease(Iterable<Integer> index, int value) {
		memory.get(index).value -= value;
	}
	
	public void increase(Iterable<Integer> index) {
		increase(index, 1);
	}
	
	public void decrease(Iterable<Integer> index) {
		decrease(index, 1);
	}
	
}


