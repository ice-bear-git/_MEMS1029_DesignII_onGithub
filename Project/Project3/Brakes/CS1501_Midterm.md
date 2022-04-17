``` Java

BinaryNode<T> findLargest(BinaryNode<T> root);
BinaryNode<T> findSmallest(BinaryNode<T> root);

public class BinaryNode<T extends Comparable<? super T>> {
	private Boolean isBST(BinaryNode<T> root) {
		Boolean result = false;
		if(root == null){
			result = true;	// The only place that can return true
		}
		else {
			/* For Left Child */
			if (root.left != null) {
				leftCmp = (root.key).CompareTo(findLargest(root.left.key));
			} else {
				leftCmp = 1;
			}

			/* For Right Child */
			if (root.right != null) {
				rightCmp = (root.key).CompareTo(findLargest(root.right.key));
			} else {
				rightCmp = -1;
			}

			/* Check for False */
			if ((leftCmp>0) && (rightCmp<0)){
				return ( (isBST(root.left)) && (isBST(root.right)) )
			}
			else {return false;}
		}
	}
} 

```