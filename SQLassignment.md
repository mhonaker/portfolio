##Introduction

For the first part of the assignment we were to work with an sqlite3 database file, consisting of a single table, fequency(docid, term, count). Each docid was the identification of a particular document, term is an English word, and count is the number of time that term appeared in the document refered to by docid. The second part of the assignment used another sqlite3 database file representing a sparse matrix where two matricies, A and B were represented as [row number, column number, value].


##Problem Set 1 - Basic Relational Algebra
a.  Write a SQL statement to select the records where the docid = 10398_txt_earn. Report the count of the records returned.

```SQL
SELECT count(*) FROM (
    SELECT *
    FROM frequency
    WHERE docid = '10398_txt_earn'
) x;
```

b. Write a SQL statement to select the records where the docid = 10398_txt_earn, and project only the set of those records that where the count of terms is one. Report the count of records returned.

```SQL
SELECT count(*) FROM (
    SELECT term
    FROM frequency
    WHERE docid = '10398_txt_earn'
    AND count = 1
) x;
```

c. Write a SQL statment to select the union of records where docid = 10398_txt_earn and docid = 925_txt_trade have terms with a count of 1. Report the count of records returned.

```SQL
SELECT count(*) FROM (
    SELECT term
    FROM frequency
    WHERE docid = '10398_txt_earn'
    AND count = 1
    UNION
    SELECT term
    FROM frequency
    WHERE docid = '925_txt_trade'
    AND count = 1
) x;
```

d. Write a SQL statement to count the number of documents containing the word "parliament".
```SQL
SELECT count(*) FROM (
    SELECT docid
    FROM frequency
    WHERE term = 'parliment'
) x;
```

e. Write a SQL statement to find all documents that have more than 300 total terms, including duplicate terms.

```SQL
SELECT count(*) FROM (
    SELECT docid
    FROM frequency
    GROUP BY docid
    HAVING sum(count) > 300
) x;
```

f. Write a SQL statement to count the number of unique documents that contain both the word 'transactions' and the word 'world'.

```SQL
SELECT count(*) FROM (
    SELECT *
    FROM frequency world, frequency trans
    WHERE world.term = 'world' AND trans.term = 'transactions'
    AND world.docid = trans.docid
) x;
```

g. Write a SQL query to compute the unnormalized similarity matrix of the document table. This is the matrix multiplied by its own transpose. Report the similiarity of documents 10080_txt_crude and 17035_txt_earn.
```SQL
SELECT sum(a.count * b.count)
FROM frequency a
JOIN frequency b
ON a.term = b.term
WHERE a.docid = '10080_txt_crude'
AND b.docid = '17035_txt_earn'; 
```

h. Find the best matching document to the keyword query "washington taxes treasury".
first:
```SQL
CREATE VIEW test1 as
    SELECT * FROM frequency
    UNION
    SELECT 'q' as docid, 'washington' as term, 1 as count
    UNION
    SELECT 'q' as docid, 'taxes' as term, 1 as count
    UNION
    SELECT 'q' as docid, 'treasury' as term, 1 as count;
    ```
then:
```SQL
SELECT docid, max(sim) from (
    SELECT f.docid, sum(f.count * q.count) as sim
    FROM frequency f
    JOIN test1 q
    ON f.term = q.term
    WHERE q.docID = 'q'
    GROUP BY f.docid
) x;
```

**Problem Set 2 - Matrix Multiplication in SQL**
a. Express A X B as a SQL query, using the matrix database.

```SQL
SELECT A.row_num, B.col_num, sum(A.value * B.value)
FROM A, B
WHERE A.col_num = B.row_num
GROUP BY A.row_num, B.col_num;
```


