#!/bin/bash
set -e

MONGO_HOST="mongod.search-community:27017"

echo "Waiting for MongoDB to be ready..."
until mongosh "mongodb://${MONGO_HOST}/" --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  echo "MongoDB is unavailable - sleeping"
  sleep 2
done
echo "MongoDB is up!"

# Check if replica set is initialized
echo "Checking replica set status..."
RS_STATUS=$(mongosh "mongodb://${MONGO_HOST}/" --quiet --eval "
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
  mongosh "mongodb://${MONGO_HOST}/" --quiet --eval "
  rs.initiate({
    _id: 'rs0',
    members: [
      { _id: 0, host: 'mongod.search-community:27017' }
    ]
  });
  "
  echo "Replica set initialized"
  
  # Wait for PRIMARY
  echo "Waiting for replica set to become PRIMARY..."
  for i in {1..30}; do
    PRIMARY_STATUS=$(mongosh "mongodb://${MONGO_HOST}/" --quiet --eval "
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
      echo "Replica set is PRIMARY"
      break
    fi
    echo "Waiting... (attempt $i/30)"
    sleep 2
  done
else
  echo "Replica set already initialized"
fi

# Create user
echo "Creating mongotUser..."
mongosh "mongodb://${MONGO_HOST}/" --quiet --eval "
const adminDb = db.getSiblingDB('admin');
try {
  adminDb.createUser({
    user: 'mongotUser',
    pwd: 'mongotPassword',
    roles: [{ role: 'searchCoordinator', db: 'admin' }]
  });
  print('User created');
} catch (error) {
  if (error.code === 51003) {
    print('User already exists');
  } else {
    print('Error: ' + error.message);
  }
}
"

echo "MongoDB setup complete!"
