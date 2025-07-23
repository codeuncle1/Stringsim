from collections import Counter
import math

class StringSimilarity:
    """A class to compute similarity scores between two strings using multiple algorithms."""

    def __init__(self):
        pass

    def levenshtein_similarity(self, str1, str2):
        """Calculate Levenshtein similarity (0 to 1)."""
        str1, str2 = str1.lower(), str2.lower()
        len1, len2 = len(str1), len(str2)
        matrix = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1] == str2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,    # deletion
                        matrix[i][j-1] + 1,    # insertion
                        matrix[i-1][j-1] + 1   # substitution
                    )

        max_length = max(len1, len2)
        if max_length == 0:
            return 1.0 if len1 == len2 else 0.0
        return 1.0 - (matrix[len1][len2] / max_length)

    def jaro_winkler_similarity(self, str1, str2, p=0.1):
        """Calculate Jaro-Winkler similarity (0 to 1)."""
        str1, str2 = str1.lower(), str2.lower()
        len1, len2 = len(str1), len(str2)

        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0

        # Maximum distance to consider characters as matching
        match_distance = max(len1, len2) // 2 - 1
        matches1 = [False] * len1
        matches2 = [False] * len2
        matches = 0
        transpositions = 0

        # Find matching characters
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if not matches2[j] and str1[i] == str2[j]:
                    matches1[i] = matches2[j] = True
                    matches += 1
                    break

        if matches == 0:
            return 0.0

        # Count transpositions
        j = 0
        for i in range(len1):
            if matches1[i]:
                while not matches2[j]:
                    j += 1
                if str1[i] != str2[j]:
                    transpositions += 1
                j += 1

        # Jaro similarity
        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

        # Winkler adjustment: boost for common prefix (up to 4 characters)
        common_prefix = 0
        for i in range(min(4, len1, len2)):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break
        jaro_winkler = jaro + common_prefix * p * (1 - jaro)

        return jaro_winkler

    def cosine_similarity(self, str1, str2):
        """Calculate Cosine similarity based on character bigrams (0 to 1)."""
        str1, str2 = str1.lower(), str2.lower()

        # Get bigrams
        def get_bigrams(s):
            return [s[i:i+2] for i in range(len(s)-1)] if len(s) > 1 else list(s)

        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)

        # Create vectors
        counter1 = Counter(bigrams1)
        counter2 = Counter(bigrams2)
        all_bigrams = set(counter1).union(set(counter2))

        vector1 = [counter1.get(bigram, 0) for bigram in all_bigrams]
        vector2 = [counter2.get(bigram, 0) for bigram in all_bigrams]

        # Calculate dot product
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vector1))
        magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vector2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def damerau_levenshtein_similarity(self, str1, str2):
        """Calculate Damerau-Levenshtein similarity (0 to 1)."""
        str1, str2 = str1.lower(), str2.lower()
        len1, len2 = len(str1), len(str2)
        matrix = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1] == str2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,    # deletion
                        matrix[i][j-1] + 1,    # insertion
                        matrix[i-1][j-1] + 1   # substitution
                    )
                    # Check for transposition
                    if i > 1 and j > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
                        matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2] + 1)

        max_length = max(len1, len2)
        if max_length == 0:
            return 1.0 if len1 == len2 else 0.0
        return 1.0 - (matrix[len1][len2] / max_length)

def compare_strings(str1, str2):
    """Compare two strings using all similarity algorithms."""
    sim = StringSimilarity()
    print(f"\nComparing: '{str1}' vs. '{str2}'")
    print(f"Levenshtein Similarity: {sim.levenshtein_similarity(str1, str2):.2%}")
    print(f"Jaro-Winkler Similarity: {sim.jaro_winkler_similarity(str1, str2):.2%}")
    print(f"Cosine Similarity (bigrams): {sim.cosine_similarity(str1, str2):.2%}")
    print(f"Damerau-Levenshtein Similarity: {sim.damerau_levenshtein_similarity(str1, str2):.2%}")
    print("-" * 50)

if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("hello", "hello"),          # Identical
        ("hello", "helo"),          # One deletion
        ("hello", "jello"),         # One substitution
        ("hello", "world"),         # Different strings
        ("", ""),                   # Empty strings
        ("python", "pythn"),       # One deletion
        ("kitten", "sitting"),      # Multiple edits
        ("listen", "silent"),       # Transposition
    ]

    for str1, str2 in test_cases:
        compare_strings(str1, str2)
