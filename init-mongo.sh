#!/bin/bash
set -e

echo "Starting MongoDB initialization..."

# Wait for MongoDB to be fully ready
echo "Waiting for MongoDB to be ready..."
until mongosh --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  echo "MongoDB is unavailable - sleeping"
  sleep 2
done
echo "MongoDB is up!"

# Initialize replica set first
echo "Checking replica set status..."
RS_STATUS=$(mongosh --quiet --eval "
try {
  rs.status();
  print('INITIALIZED');
} catch (error) {
  if (error.code === 94 || error.message.includes('no replset config')) {
    print('NOT_INITIALIZED');
  } else {
    print('ERROR: ' + error.message);
  }
}
" | tail -1)

if [ "$RS_STATUS" = "NOT_INITIALIZED" ]; then
  echo "Initializing replica set..."
  mongosh --eval "
  rs.initiate({
    _id: 'rs0',
    members: [
      { _id: 0, host: 'mongod.search-community:27017' }
    ]
  });
  print('Replica set initialized successfully');
  "
  
  # Wait for replica set to become ready
  echo "Waiting for replica set to become PRIMARY..."
  for i in {1..30}; do
    PRIMARY_STATUS=$(mongosh --quiet --eval "
    try {
      const status = rs.status();
      if (status.myState === 1) {
        print('PRIMARY');
      } else {
        print('NOT_PRIMARY');
      }
    } catch (e) {
      print('ERROR');
    }
    " | tail -1)
    
    if [ "$PRIMARY_STATUS" = "PRIMARY" ]; then
      echo "Replica set is ready and PRIMARY"
      break
    fi
    echo "Waiting... (attempt $i/30)"
    sleep 2
  done
else
  echo "Replica set already initialized: $RS_STATUS"
fi

# Create user using local connection (no port specification needed)
echo "Creating user..."
mongosh --eval "
const adminDb = db.getSiblingDB('admin');
try {
adminDb.createUser({
   user: 'mongotUser',
   pwd: 'mongotPassword',
   roles: [{ role: 'searchCoordinator', db: 'admin' }]
});
print('User mongotUser created successfully');
} catch (error) {
if (error.code === 11000) {
   print('User mongotUser already exists');
} else {
   print('Error creating user: ' + error);
}
}
"

echo "MongoDB initialization completed."