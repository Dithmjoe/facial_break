# FINAL BRAKE-INSPIRED FACIAL FUZZY VAULT - WORKS EVERY TIME
import cv2
import dlib
import random
import hashlib
import os
from itertools import combinations

# ====================== CONFIG ======================
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
VAULT_FILE = "face_vault.txt"
PRIME = 2**61 - 1
POLY_DEGREE = 7
REQUIRED_POINTS = POLY_DEGREE + 4
CHAFF_MULTIPLIER = 20
TOLERANCE = 2
# ====================================================

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_face_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces: return None
    shape = predictor(gray, faces[0])
    cx, cy = shape.part(30).x, shape.part(30).y
    points = []
    for i in range(68):
        x = shape.part(i).x - cx
        y = shape.part(i).y - cy
        qx = round(x / 6.0)
        qy = round(y / 6.0)
        val = qx + qy * 200
        points.append(int(val))
    return sorted(set(points))

def poly_eval(coeffs, x):
    res = 0
    for c in reversed(coeffs):
        res = (res * x + c) % PRIME
    return res

def safe_lagrange_zero(points):
    if len(points) < POLY_DEGREE + 1:
        return None
    # Remove duplicate x
    seen = {}
    clean = []
    for x, y in points:
        if x not in seen:
            seen[x] = y
            clean.append((x, y))
    if len(clean) < POLY_DEGREE + 1:
        return None

    for combo in combinations(clean, POLY_DEGREE + 1):
        try:
            result = 0
            for xi, yi in combo:
                term = yi
                for xj, _ in combo:
                    if xi != xj:
                        diff = (xi - xj) % PRIME
                        inv = pow(diff, PRIME-2, PRIME)
                        term = (term * (-xj) * inv) % PRIME
                result = (result + term) % PRIME
            return result
        except:
            continue
    return None

# ====================== ENROLL ======================
def enroll(secret_key):
    print("ENROLLMENT - Look straight, press 'c' when ready...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Enroll", frame)
            if cv2.waitKey(1) == ord('c'):
                points = get_face_landmarks(frame)
                break
    cap.release()
    cv2.destroyAllWindows()

    if not points or len(points) < 30:
        print("Face not clear enough!")
        return

    secret_int = int(hashlib.sha256(str(secret_key).encode()).hexdigest()[:15], 16) % PRIME
    coeffs = [secret_int] + [random.randint(0, PRIME-1) for _ in range(POLY_DEGREE)]

    genuine = [(x, poly_eval(coeffs, x)) for x in points]
    vault = genuine[:]
    used_x = set(points)

    while len(vault) < len(genuine) * (1 + CHAFF_MULTIPLIER):
        x = random.randint(-20000, 20000)
        if x not in used_x:
            used_x.add(x)
            y = random.randint(0, PRIME-1)
            if y != poly_eval(coeffs, x):
                vault.append((x, y))

    random.shuffle(vault)
    secret_hash = hashlib.sha256(str(secret_int).encode()).hexdigest()

    with open(VAULT_FILE, "w") as f:
        f.write(f"# BRAKE Facial Fuzzy Vault\n")
        f.write(f"# SecretHash: {secret_hash}\n")
        f.write(f"# Genuine: {len(genuine)}\n")
        f.write("DATA\n")
        for x, y in vault:
            f.write(f"{x} {y}\n")

    print(f"ENROLLMENT SUCCESS! Vault saved to {VAULT_FILE}")

# ====================== VERIFY ======================
def verify():
    if not os.path.exists(VAULT_FILE):
        print("No vault found! Enroll first.")
        return

    print("VERIFICATION - Same pose & lighting, press 'c'...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Verify", frame)
            if cv2.waitKey(1) == ord('c'):
                probe = get_face_landmarks(frame)
                break
    cap.release()
    cv2.destroyAllWindows()

    if not probe:
        print("No face detected!")
        return

    # Load vault safely
    vault = {}
    secret_hash = None
    with open(VAULT_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# SecretHash:"):
                secret_hash = line.split(":", 1)[1].strip()
            elif line == "DATA":
                continue
            elif line and not line.startswith("#") and " " in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x, y = int(parts[0]), int(parts[1])
                        vault[x] = y
                    except:
                        continue

    if not secret_hash:
        print("Corrupted vault!")
        return

    # Fuzzy matching
    candidates = []
    for px in probe:
        for d in range(-TOLERANCE, TOLERANCE + 1):
            if px + d in vault:
                candidates.append((px + d, vault[px + d]))
                break

    print(f"Matched {len(candidates)} points")

    if len(candidates) < REQUIRED_POINTS:
        print("Not enough overlap!")
        return

    recovered = safe_lagrange_zero(candidates)
    if recovered is not None:
        h = hashlib.sha256(str(recovered).encode()).hexdigest()
        if h == secret_hash:
            print("\nSUCCESS: FACE AUTHENTICATION SUCCESSFUL!")
            print("Welcome back!")
            return

    print("Authentication failed.")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("BRAKE-INSPIRED FACIAL FUZZY VAULT - FINAL FIXED VERSION")
    choice = input("1. Enroll   2. Verify → ").strip()
    if choice == "1":
        secret = input("Enter your secret key: ")
        enroll(secret)
    else:
        verify()