function initializeFirebase() {
  const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
    messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
    appId: "YOUR_APP_ID",
    measurementId: "YOUR_MEASUREMENT_ID"
  };

  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
}

function logToFirebase(data) {
  const db = firebase.firestore();
  db.collection("trap_logs").add(data)
    .then(() => {
      console.log("✅ Log added to Firebase");
    })
    .catch((error) => {
      console.error("❌ Error adding log: ", error);
    });
}
