import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# 1️⃣ استيراد البيانات
# ----------------------------
df = pd.read_excel("Customer_Behavior_Large.xlsx")
print("أول 5 صفوف من البيانات:")
print(df.head())

# معلومات عن البيانات
print("\nمعلومات عن البيانات:")
print(df.info())
print("\nالإحصاءات الوصفية:")
print(df.describe())

# التحقق من القيم الفارغة
print("\nالقيم الفارغة لكل عمود:")
print(df.isnull().sum())

# ----------------------------
# 2️⃣ التحليل الديموغرافي
# ----------------------------
# توزيع الجنس
plt.figure(figsize=(6,4))
df['Gender'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()

# توزيع الأعمار
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=12, color='lightgreen', edgecolor='black')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

# أكثر المدن شيوعاً
print("\nأكثر 10 مدن شيوعاً:")
print(df['City'].value_counts().head(10))

# ----------------------------
# 3️⃣ تحليل سلوك العملاء
# ----------------------------
# أعلى العملاء زيارة
top_visits = df.sort_values(by='Visits', ascending=False).head(10)
print("\nأعلى 10 عملاء زيارة:")
print(top_visits[['CustomerID','Name','Visits']])

# أعلى العملاء إنفاقاً
top_spent = df.sort_values(by='TotalSpent', ascending=False).head(10)
print("\nأعلى 10 عملاء إنفاقاً:")
print(top_spent[['CustomerID','Name','TotalSpent']])

# متوسط الإنفاق حسب الفئة المفضلة
category_spent = df.groupby('FavoriteCategory')['TotalSpent'].mean().sort_values(ascending=False)
print("\nمتوسط الإنفاق حسب الفئة المفضلة:")
print(category_spent)

# ----------------------------
# 4️⃣ تقسيم العملاء إلى شرائح (Segmentation)
# ----------------------------
X = df[['Age', 'Visits', 'TotalSpent']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# استخدام K-Means لتقسيم العملاء إلى 4 شرائح
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(X_scaled)

print("\nعدد العملاء في كل شريحة:")
print(df['Segment'].value_counts())

# تحليل كل شريحة
print("\nمتوسط العمر، الزيارات، والإنفاق لكل شريحة:")
print(df.groupby('Segment')[['Age','Visits','TotalSpent']].mean())

# ----------------------------
# 5️⃣ المخططات البيانية للشرائح
# ----------------------------
plt.figure(figsize=(8,5))
for seg in df['Segment'].unique():
    subset = df[df['Segment']==seg]
    plt.scatter(subset['Visits'], subset['TotalSpent'], label=f'Segment {seg}', alpha=0.5)
plt.title("Customer Segmentation: Visits vs TotalSpent")
plt.xlabel("Visits")
plt.ylabel("TotalSpent")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
df.groupby('Segment')['Age'].mean().plot(kind='bar', color='orange', edgecolor='black')
plt.title("Average Age per Segment")
plt.xlabel("Segment")
plt.ylabel("Average Age")
plt.show()

plt.figure(figsize=(8,5))
df.groupby('Segment')['TotalSpent'].mean().plot(kind='bar', color='purple', edgecolor='black')
plt.title("Average TotalSpent per Segment")
plt.xlabel("Segment")
plt.ylabel("Average TotalSpent")
plt.show()

# ----------------------------
# 6️⃣ حفظ نسخة من البيانات مع الشرائح
# ----------------------------
df.to_excel("Downloads\Customer_Behavior_Segmented.xlsx", index=False)
print("\nتم حفظ الملف Customer_Behavior_Segmented.xlsx مع إضافة عمود الشرائح.")
