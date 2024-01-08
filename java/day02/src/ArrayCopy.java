public class ArrayCopy {
	
	public static void main(String[] args) {
		int [] a = {4,5,1,2,9};
		int [] b = new int[3];
		
		System.arraycopy(a, 2, b, 0, 3);
		
		for (int val:b)
			System.out.print(val+" ");

	}

}
